#ifndef __UTIL_HPP
#define __UTIL_HPP

#include <vector>
#include <string>
#include <type_traits>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <functional>
#include <unordered_map>
#include <map>
#include <set>
#include <unordered_set>
#include <fcntl.h>
#include <unistd.h>
#include <random>

namespace string {

    /***
     * split string by the specified delimiter, if delim is the empty string
     * then a vector of chars is returned
     * @param str the string to split
     * @param delim the delimiter, by default a single space
     * @return a vector of strings
     */
    inline std::vector<std::string> split(const std::string &str, const std::string &delim = " ") {
        if (delim.empty()) {
            throw std::runtime_error("empty delimiter");
        }
        std::vector<std::string> tokens;
        size_t last = 0;
        size_t next = 0;
        while ((next = str.find(delim, last)) != std::string::npos) {
            tokens.emplace_back(str.substr(last, next - last));
            last = next + 1;
        }
        tokens.emplace_back(str.substr(last));
        return tokens;
    }

    /***
     * check if str1 starts with str2
     * @param str1
     * @param str2
     * @return
     */
    inline bool starts_with(const std::string &str1, const std::string &str2) {
        return str1.size() >= str2.size() && std::equal(str1.begin(), str1.begin() + str2.size(), str2.begin());
    }

    /***
     * check if str1 end with str2
     * @param str1
     * @param str2
     * @return
     */
    inline bool ends_with(const std::string &str1, const std::string &str2) {
        return str1.size() >= str2.size() && std::equal(str1.end() - str2.size(), str1.end(), str2.begin());
    }

    /***
     * equals ignore case
     * @param str1
     * @param str2
     * @return
     */
    inline bool iequals(const std::string &str1, const std::string &str2) {
        return str1.size() == str2.size() &&
               std::equal(str1.begin(), str1.end(), str2.begin(), [](char a, char b) -> bool {
                   return std::tolower(a) == std::tolower(b);
               });
    }

    inline bool contains(const std::string &str1, const std::string &str2) {
        return str1.size() <= str2.size() && str1.find_first_of(str2, 0) != std::string::npos;
    }

    /***
     * cast string to arithmetic type specified by the template argument
     * @tparam T type to case to
     * @param str string to be casted
     * @return arithmetic type
     */
    template<typename T>
    constexpr T to(const std::string &str) {
        static_assert(std::is_fundamental<T>::value && (std::is_arithmetic<T>::value || std::is_same<T, bool>::value),
                      "cannot convert std::string to non-arithmetic type");

        // signed types
        if (std::is_same<T, char>::value || std::is_same<T, short>::value || std::is_same<T, int>::value ||
            std::is_same<T, long>::value || std::is_same<T, long long>::value) {
            return (T) std::strtoll(str.c_str(), nullptr, 10);
        } // unsigned types
        else if (std::is_same<T, unsigned char>::value || std::is_same<T, unsigned short>::value ||
                 std::is_same<T, unsigned int>::value ||
                 std::is_same<T, unsigned long>::value || std::is_same<T, unsigned long long>::value) {
            return (T) std::strtoull(str.c_str(), nullptr, 10);
        } // double
        else if (std::is_same<T, double>::value) {
            return std::strtod(str.c_str(), nullptr);
        } // float
        else if (std::is_same<T, float>::value) {
            return std::strtof(str.c_str(), nullptr);
        } // long double
        else if (std::is_same<T, long double>::value) {
            return std::strtold(str.c_str(), nullptr);
        } // bool
        else if (std::is_same<T, bool>::value) {
            if (str == "true" || str == "1" || str == "True" || str == "TRUE")
                return true;
            else if (str == "false" || str == "0" || str == "False" || str == "FALSE")
                return false;
            else
                throw std::runtime_error("cannot cast std::string '" + str + "' to bool");
        } else {
            throw std::domain_error("cannot convert std::string to specified type");
        }
    }

    /***
     * remove leading and trailing whitespace characters (can be any characters) from string
     * @param str
     * @param whitespace
     * @return
     */
    inline std::string trim(const std::string &str, const std::string &whitespace = " \t\n\r") {
        const size_t begin = str.find_first_not_of(whitespace);
        const size_t end = str.find_last_not_of(whitespace);
        return str.substr(begin, end - begin + 1);
    }

    /***
     * convert string to upper case
     * @param str
     * @return
     */
    inline std::string to_upper(const std::string &str) {
        std::string up = str;
        std::transform(str.begin(), str.end(), up.begin(), ::toupper);
        return up;
    }

    /***
     * convert string to lower case
     * @param str
     * @return
     */
    inline std::string to_lower(const std::string &str) {
        std::string lo = str;
        std::transform(str.begin(), str.end(), lo.begin(), ::tolower);
        return lo;
    }
}

namespace seq {

    template<typename iterator>
    inline decltype(iterator::operator*) min(iterator begin, iterator end) {
        if (end - begin == 0) {
            throw std::length_error("empty sequence");
        }

        iterator min = begin;
        for (auto it = begin; it != end; ++it) {
            min = *it < *min ? it : min;
        }
        return *min;
    }

    template<typename iterator>
    inline decltype(iterator::operator*) max(iterator begin, iterator end) {
        if (end - begin == 0) {
            throw std::length_error("empty sequence");
        }

        iterator max = begin;
        for (auto it = begin; it != end; ++it) {
            max = *it > *max ? it : max;
        }
        return *max;
    }

    template<typename iterator>
    inline uint64_t argmin(iterator begin, iterator end) {
        if (end - begin == 0) {
            throw std::length_error("empty sequence");
        }

        iterator min = begin;
        for (auto it = begin; it != end; ++it) {
            min = *it < *min ? it : min;
        }
        return min - begin;
    }

    template<typename iterator>
    inline uint64_t argmax(iterator begin, iterator end) {
        if (end - begin == 0) {
            throw std::length_error("empty sequence");
        }

        iterator max = begin;
        for (auto it = begin; it != end; ++it) {
            max = *it > *max ? it : max;
        }
        return max - begin;
    }
}

namespace io {
    /***
     * generalized STL-like print function
     * @tparam iterator
     * @param os
     * @param begin
     * @param end
     * @param b
     * @param e
     * @param print_func
     */
    template <typename iterator>
    inline void print(std::ostream &os, iterator begin, iterator end, char b, char e,
    std::function<void (std::ostream &os, iterator it)> &print_func){
    auto it = begin;
    os << b;
    while (it != end) {
        print_func(os, it);
        auto _it = it;
        if (++_it != end) {
            os << ", ";
        }
    }
    os << e;
}

template <typename iterator>
inline void print(std::ostream &os, iterator begin, iterator end, char b='(', char e=')'){
    auto it = begin;
    os << b;
    while (it != end) {
        os << *it;
        auto _it = it;
        if (++_it != end) {
            os << ", ";
        }
        ++it;
    }
    os << e;
}
}

template <typename T>
inline std::ostream& operator<<(std::ostream &os, const std::vector<T> &vec) {
    io::print(os, vec.begin(), vec.end(), '[', ']');
    return os;
}

template <typename T, size_t N>
inline std::ostream& operator<<(std::ostream &os, const std::array<T, N> &array) {
    io::print(os, array.begin(), array.end(), '[', ']');
    return os;
}

template <typename T>
inline std::ostream& operator<<(std::ostream &os, const std::set<T> &set) {
    io::print(os, set.begin(), set.end(), '{', '}');
    return os;
}

template <typename T>
inline std::ostream& operator<<(std::ostream &os, const std::unordered_set<T> &set) {
    io::print(os, set.begin(), set.end(), '{', '}');
    return os;
}

template <typename K, typename V>
inline std::ostream& operator<<(std::ostream &os, const std::unordered_map<K, V> &map) {
    io::print(os, map.begin(), map.end(), '{', '}', [](std::ostream &os, decltype(map.begin()) it) { os << it->first << ": " << it->second; });
    return os;
}

template <typename K, typename V>
inline std::ostream& operator<<(std::ostream &os, const std::map<K, V> &map) {
    io::print(os, map.begin(), map.end(), '{', '}', [](std::ostream &os, decltype(map.begin()) it) { os << it->first << ": " << it->second; });
    return os;
}

/***
 * byte swap using gcc built-ins for various interger types
 * @tparam T integer type
 * @param x value to byte swap
 * @return
 */
template<typename T>
inline T bswap(T x) {
    if (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
        return (T) x;
    } else if (std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value) {
        return (T) __bswap_16((uint16_t) x);
    } else if (std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value) {
        return (T) __bswap_32((uint32_t) x);
    } else if (std::is_same<T, int64_t>::value || std::is_same<T, uint64_t>::value) {
        return (T) __bswap_64((uint64_t) x);
    } else {
        return (T) x;
    }
}

/***
 * convert data type to network byte order and back
 * @tparam T
 * @param x
 * @return
 */
template <typename T>
inline T inet_bswap(T x) {
#if BYTE_ORDER == LITTLE_ENDIAN
    return bswap<T>(x);
#else
    return (T) x;
#endif
}

inline bool exists(const std::string& fname) {
    return (access(fname.c_str(), F_OK ) != -1);
}

#endif // __UTIL_HPP
