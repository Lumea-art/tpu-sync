// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tpu_sync/core/raw_transfer_core.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_device_event.h"
#include "xla/pjrt/c/pjrt_c_api_raw_buffer_extension.h"
#include "tpu_sync/core/tpu_utils.h"

namespace raiden {
namespace {

absl::Status PjrtErrorToStatusWithCode(const PJRT_Api* c_api,
                                       PJRT_Error* error) {
  if (error == nullptr) {
    return absl::OkStatus();
  }
  if (c_api == nullptr) {
    return absl::InternalError(
        "PJRT external DMA returned PJRT_Error without PJRT_Api");
  }
  std::string message;
  if (c_api->PJRT_Error_Message != nullptr) {
    PJRT_Error_Message_Args msg_args;
    msg_args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    msg_args.extension_start = nullptr;
    msg_args.error = error;
    msg_args.message = nullptr;
    msg_args.message_size = 0;
    c_api->PJRT_Error_Message(&msg_args);
    if (msg_args.message != nullptr) {
      message.assign(msg_args.message, msg_args.message_size);
    }
  }
  absl::StatusCode code = absl::StatusCode::kInternal;
  if (c_api->PJRT_Error_GetCode != nullptr) {
    PJRT_Error_GetCode_Args code_args;
    code_args.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
    code_args.extension_start = nullptr;
    code_args.error = error;
    code_args.code = PJRT_Error_Code_INTERNAL;
    PJRT_Error* code_err = c_api->PJRT_Error_GetCode(&code_args);
    if (code_err != nullptr) {
      if (c_api->PJRT_Error_Destroy != nullptr) {
        PJRT_Error_Destroy_Args destroy_code_err;
        destroy_code_err.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
        destroy_code_err.extension_start = nullptr;
        destroy_code_err.error = code_err;
        c_api->PJRT_Error_Destroy(&destroy_code_err);
      }
    } else if (code_args.code != PJRT_Error_Code_OK) {
      code = static_cast<absl::StatusCode>(code_args.code);
    }
  }
  if (c_api->PJRT_Error_Destroy != nullptr) {
    PJRT_Error_Destroy_Args destroy_args;
    destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroy_args.extension_start = nullptr;
    destroy_args.error = error;
    c_api->PJRT_Error_Destroy(&destroy_args);
  }
  return absl::Status(code, message);
}

template <typename FnPtr>
struct SecondParamPointee;

template <typename Ret, typename Arg0, typename Arg1>
struct SecondParamPointee<Ret (*)(Arg0, Arg1*)> {
  using type = Arg1;
};

template <typename TableT, typename = void>
struct ExternalDmaAccessor {
  static absl::StatusOr<std::shared_ptr<const ExternalDmaInfo>> Acquire(
      PJRT_RawBuffer* /*raw_buffer*/, const PJRT_Api* /*c_api*/) {
    return absl::UnimplementedError(
        "AcquireExternalDmaAccess is not supported in this XLA version");
  }
};

template <typename TableT>
struct ExternalDmaAccessor<
    TableT, std::void_t<decltype(&TableT::acquire_external_dma_access)>> {
  static absl::StatusOr<std::shared_ptr<const ExternalDmaInfo>> Acquire(
      PJRT_RawBuffer* raw_buffer, const PJRT_Api* c_api) {
    using AccessT = typename SecondParamPointee<
        decltype(TableT::acquire_external_dma_access)>::type;
    if (raw_buffer == nullptr || raw_buffer->vtable == nullptr) {
      return absl::UnimplementedError(
          "AcquireExternalDmaAccess: no PJRT_RawBuffer attached to "
          "RaidenBufferHandle");
    }
    const TableT* vtable = raw_buffer->vtable;
    const size_t required_size = offsetof(TableT, acquire_external_dma_access) +
                                 sizeof(vtable->acquire_external_dma_access);
    if (vtable->struct_size < required_size ||
        vtable->acquire_external_dma_access == nullptr) {
      return absl::UnimplementedError(
          "AcquireExternalDmaAccess: PJRT plugin does not implement "
          "acquire_external_dma_access");
    }

    AccessT c_access = {nullptr, nullptr};
    PJRT_Error* acquire_err =
        vtable->acquire_external_dma_access(raw_buffer, &c_access);
    if (acquire_err != nullptr) {
      return PjrtErrorToStatusWithCode(c_api, acquire_err);
    }
    if (c_access.vtable == nullptr || c_access.vtable->destroy == nullptr ||
        c_access.vtable->get_event == nullptr ||
        c_access.vtable->get_info == nullptr) {
      if (c_access.vtable != nullptr && c_access.vtable->destroy != nullptr) {
        c_access.vtable->destroy(&c_access);
      }
      return absl::InternalError(
          "AcquireExternalDmaAccess: plugin returned incomplete "
          "PJRT_ExternalDmaAccess vtable");
    }

    PJRT_DeviceEvent event = c_access.vtable->get_event(&c_access);
    if (event.vtable == nullptr) {
      c_access.vtable->destroy(&c_access);
      return absl::InternalError(
          "AcquireExternalDmaAccess: null PJRT_DeviceEvent vtable");
    }
    if (event.vtable->get_state(event.device_event) ==
        PJRT_DeviceEvent_State_Unavailable) {
      absl::Notification ready;
      event.vtable->and_then(
          event.device_event,
          +[](void* user_arg) {
            static_cast<absl::Notification*>(user_arg)->Notify();
          },
          &ready);
      ready.WaitForNotification();
    }

    PJRT_Error_Code err_code = PJRT_Error_Code_OK;
    const char* err_msg = nullptr;
    size_t err_msg_size = 0;
    if (event.vtable->get_error_if_present(event.device_event, &err_code,
                                           &err_msg, &err_msg_size) != 0) {
      absl::Status status(
          err_code == PJRT_Error_Code_OK
              ? absl::StatusCode::kInternal
              : static_cast<absl::StatusCode>(err_code),
          err_msg != nullptr ? absl::string_view(err_msg, err_msg_size) : "");
      c_access.vtable->destroy(&c_access);
      return status;
    }

    int fd = -1;
    uint64_t offset = 0;
    uint64_t size = 0;
    PJRT_Error* info_err =
        c_access.vtable->get_info(&c_access, &fd, &offset, &size);
    if (info_err != nullptr) {
      absl::Status status = PjrtErrorToStatusWithCode(c_api, info_err);
      c_access.vtable->destroy(&c_access);
      return status;
    }
    if (fd < 0 || size == 0 || offset > std::numeric_limits<size_t>::max() ||
        size > std::numeric_limits<size_t>::max()) {
      c_access.vtable->destroy(&c_access);
      return absl::InvalidArgumentError(absl::StrCat(
          "AcquireExternalDmaAccess returned invalid dma-buf info: fd=", fd,
          " offset=", offset, " size=", size));
    }

    return std::shared_ptr<const ExternalDmaInfo>(
        new ExternalDmaInfo{
            .dma_buf_fd = fd,
            .offset = static_cast<size_t>(offset),
            .size = static_cast<size_t>(size),
        },
        [c_access](const ExternalDmaInfo* p) mutable {
          c_access.vtable->destroy(&c_access);
          delete p;
        });
  }
};

}  // namespace

absl::StatusOr<std::shared_ptr<const ExternalDmaInfo>>
RaidenBufferHandle::AcquireExternalDmaAccess() const {
  PJRT_RawBuffer* raw_buffer = nullptr;
  const PJRT_Api* c_api = nullptr;
  if (is_common_buffer) {
#if RAIDEN_JAX >= 1002
    raw_buffer = common_raw_buffer.get();
#endif
  } else {
    raw_buffer = c_raw_buffer;
    c_api = c_hold != nullptr ? c_hold->c_api : nullptr;
  }
  return ExternalDmaAccessor<PJRT_RawBuffer_FunctionTable>::Acquire(raw_buffer,
                                                                    c_api);
}

std::vector<int> DetectNumaNodes(
    const std::vector<std::vector<RaidenBufferHandle>>& layer_buffers) {
  std::vector<int> unique_numa_nodes;
  for (const auto& layer : layer_buffers) {
    for (const auto& buf : layer) {
      if (buf.device) {
        int node = tpu_raiden::GetPjRtDeviceNumaNode(buf.device);
        if (node >= 0) {
          bool found = false;
          for (int n : unique_numa_nodes) {
            if (n == node) {
              found = true;
              break;
            }
          }
          if (!found) unique_numa_nodes.push_back(node);
        }
      }
    }
  }
  return unique_numa_nodes;
}

std::vector<tpu_raiden::transport::ShardInfo> DetectShards(
    const std::vector<std::vector<RaidenBufferHandle>>& layer_buffers) {
  std::vector<tpu_raiden::transport::ShardInfo> shards;
  if (layer_buffers.empty()) {
    return shards;
  }
  // Every layer places shard `i` on the same device, so layer 0 is
  // representative.
  shards.reserve(layer_buffers[0].size());
  for (size_t sh = 0; sh < layer_buffers[0].size(); ++sh) {
    const RaidenBufferHandle& buf = layer_buffers[0][sh];
    int node = -1;
    if (buf.device) {
      node = tpu_raiden::GetPjRtDeviceNumaNode(buf.device);
    }
    shards.push_back({.local_index = static_cast<int>(sh),
                      .numa_node = node >= 0 ? node : -1});
  }
  return shards;
}

}  // namespace raiden
