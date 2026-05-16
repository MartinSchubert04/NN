#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using int8_t = int8_t;
using int16_t = int16_t;
using int32_t = int32_t;
using int64_t = int64_t;

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using f32 = float;

using b8 = int8_t;
using b32 = int32_t;

// template <typename T>
// using Scope = std::unique_ptr<T>;
// template <typename T, typename... Args>
// constexpr Scope<T> createScope(Args &&...args) {
//   return std::make_unique<T>(std::forward<Args>(args)...);
// }

#define DATA_PATH "NN/data/"

template <typename T>
using Ref = std::shared_ptr<T>;
template <typename T, typename... Args>
constexpr Ref<T> createRef(Args &&...args) {
  return std::make_shared<T>(std::forward<Args>(args)...);
}

template <typename T>
std::string toString(T val) {
  return std::to_string(val);
}

#define LOG " [" + __FILE__ + ":" + std::to_string(__LINE__) + "]"
