#ifndef PY_MMVII_TYPECONV_H
#define PY_MMVII_TYPECONV_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "pybind11/detail/common.h"

#include "MMVII_Ptxd.h"

/*
 *  Custom type casters for some MMVVI types
 *  See: <pybind11/stl.h>
 *
 */


// Warning: functions using non-const Ptxd ref must be explicitly fixed for python
// regex to find them: "[^t ] *cPt.d. *\&"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

// cPtxd <-> array(Type)

template <typename Type, typename Value, int Dim>
struct ptxd_caster {
    using value_conv = make_caster<Value>;

    bool load(handle src, bool convert) {
        if (!isinstance<sequence>(src) || isinstance<bytes>(src) || isinstance<str>(src)) {
            return false;
        }
        auto s = reinterpret_borrow<sequence>(src);
        if (s.size() != Dim)
            return false;
        int i = 0;
        for (auto it : s) {
            value_conv conv;
            if (!conv.load(it, convert)) {
                return false;
            }
            value[i++] = cast_op<Value &&>(std::move(conv));
        }
        return true;
    }

    template <typename T>
    static handle cast(T &&src, return_value_policy policy, handle parent) {
        if (!std::is_lvalue_reference<T>::value) {
            policy = return_value_policy_override<Value>::policy(policy);
        }
        array_t<Value> a(Dim);
        for (int i=0; i<Dim; i++) {
            auto value_ = reinterpret_steal<object>(
                value_conv::cast(detail::forward_like<T>(src[i]), policy, parent));
            if (!value_) {
                return handle();
            }
            try {
                a.mutable_at(i) = value_.release().template cast<Value>();
            } catch(cast_error&) {
                return handle();
            }
        }
        return a.release();
    }

    PYBIND11_TYPE_CASTER(Type, const_name("array[") + value_conv::name + const_name("]"));
};

template <typename Type, int Dim>
struct type_caster<MMVII::cPtxd<Type, Dim>> : ptxd_caster<MMVII::cPtxd<Type, Dim>, Type, Dim> {};


PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)


#endif // PY_MMVII_TYPECONV_H
