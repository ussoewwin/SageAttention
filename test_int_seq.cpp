
#include <cstddef>
#include <utility>

namespace cute {
template <class T, class = void> struct tuple_size;
template <std::size_t I, class T, class = void> struct tuple_element;

template <class T, T... Ints>
struct tuple_size<std::integer_sequence<T, Ints...>> {
    static constexpr std::size_t value = sizeof...(Ints);
};

template <std::size_t I, class T, T... Is>
struct tuple_element<I, std::integer_sequence<T, Is...>> {
};
}

int main() {
    return 0;
}
