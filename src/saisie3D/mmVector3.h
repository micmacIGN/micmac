#ifndef MM_VECTOR3_H
#define MM_VECTOR3_H

//system
#include <math.h>

class Vector3
{
public:

    union
    {
        struct
        {
            float x,y,z;
        };
        float u[3];
    };

    //! Default constructor
    /** Inits vector to (0,0,0).
    **/
    inline Vector3(float s=0.) :x(s),y(s),z(s) {}
    //! Constructor from a triplet of coordinates
    /** Inits vector to (x,y,z).
    **/
    inline Vector3(float _x, float _y, float _z) :x(_x),y(_y),z(_z) {}
    //! Constructor from an array of 3 elements
    inline Vector3(const float p[]) :x(p[0]),y(p[1]),z(p[2]) {}
    //! Copy constructor
    inline Vector3(const Vector3& v) :x(v.x),y(v.y),z(v.z) {}

    //! Dot product
    inline float dot(const Vector3& v) const {return (x*v.x)+(y*v.y)+(z*v.z);}
    //! Cross product
    inline Vector3 cross(const Vector3 &v) const {return Vector3((y*v.z)-(z*v.y), (z*v.x)-(x*v.z), (x*v.y)-(y*v.x));}
    //! Returns vector square norm
    inline float norm2() const {return (x*x)+(y*y)+(z*z);}
    //! Returns vector norm
    inline float norm() const {return sqrt(norm2());}
    //! Sets vector norm to unity
    inline void normalize() {float n = norm2(); if (n>0.0f) *this /= sqrt(n);}
    //! Returns a normalized vector which is orthogonal to this one
    inline Vector3 orthogonal() const {Vector3 ort; vorthogonal(u, ort.u); return ort;}

    //! Inverse operator
    inline Vector3& operator - () {x=-x; y=-y; z=-z; return *this;}
    //! In-place addition operator
    inline Vector3& operator += (const Vector3& v) {x+=v.x; y+=v.y; z+=v.z; return *this;}
    //! In-place substraction operator
    inline Vector3& operator -= (const Vector3& v) {x-=v.x; y-=v.y; z-=v.z; return *this;}
    //! In-place multiplication (by a scalar) operator
    inline Vector3& operator *= (float v) {x*=v; y*=v; z*=v; return *this;}
    //! In-place division (by a scalar) operator
    inline Vector3& operator /= (float v) {x/=v; y/=v; z/=v; return *this;}
    //! Addition operator
    inline Vector3 operator + (const Vector3& v) const {return Vector3(x+v.x, y+v.y, z+v.z);}
    //! Substraction operator
    inline Vector3 operator - (const Vector3& v) const {return Vector3(x-v.x, y-v.y, z-v.z);}
    //! Multiplication operator
    inline Vector3 operator * (float s) const {return Vector3(x*s, y*s, z*s);}
    //! Division operator
    inline Vector3 operator / (float s) const {return Vector3(x/s, y/s, z/s);}
    //! Cross product operator
    inline Vector3 operator * (const Vector3& v) const {return cross(v);}
    //! Copy operator
    inline Vector3& operator = (const Vector3 &v) {x=v.x; y=v.y; z=v.z; return *this;}
    //! Dot product operator
    inline float operator && (const Vector3 &v) const {return dot(v);}
    //! Direct coordinate access
    inline float& operator [] (unsigned i) {return u[i];}
    //! Direct coordinate access (const)
    inline const float& operator [] (unsigned i) const {return u[i];}
    //! Multiplication by a scalar (front) operator
    friend Vector3 operator * (float s, const Vector3 &v);

    template<class Type> static inline Type vdot(const Type p[], const Type q[]) {return (p[0]*q[0])+(p[1]*q[1])+(p[2]*q[2]);}
    template<class Type> static inline void vcross(const Type p[], const Type q[], Type r[]) {r[0]=(p[1]*q[2])-(p[2]*q[1]); r[1]=(p[2]*q[0])-(p[0]*q[2]); r[2]=(p[0]*q[1])-(p[1]*q[0]);}
    template<class Type> static inline void vcopy(const Type p[], Type q[]) {q[0]=p[0]; q[1]=p[1]; q[2]=p[2];}
    template<class Type> static inline void vset(Type p[], Type s) {p[0]=p[1]=p[2]=s;}
    template<class Type> static inline void vset(Type p[], Type x, Type y, Type z) {p[0]=x; p[1]=y; p[2]=z;}
    template<class Type> static inline void vmultiply(const Type p[], Type s, Type r[]) {r[0]=p[0]*s; r[1]=p[1]*s; r[2]=p[2]*s;}
    template<class Type> static inline void vmultiply(Type p[], Type s) {p[0]*=s; p[1]*=s; p[2]*=s;}
    template<class Type> static inline void vadd(const Type p[], const Type q[], Type r[]) {r[0]=p[0]+q[0]; r[1]=p[1]+q[1]; r[2]=p[2]+q[2];}
    template<class Type> static inline void vsubstract(const Type p[], const Type q[], Type r[]) {r[0]=p[0]-q[0]; r[1]=p[1]-q[1]; r[2]=p[2]-q[2];}
    template<class Type> static inline void vcombination(Type a, const Type p[], Type b, const Type q[], Type r[]) {r[0]=(a*p[0])+(b*q[0]); r[1]=(a*p[1])+(b*q[1]); r[2]=(a*p[2])+(b*q[2]);}
    template<class Type> static inline void vcombination(const Type p[], Type b, const Type q[], Type r[]) {r[0]=p[0]+(b*q[0]); r[1]=p[1]+(b*q[1]); r[2]=p[2]+(b*q[2]);}
    template<class Type> static inline float vnorm2(const Type p[]) {return (p[0]*p[0])+(p[1]*p[1])+(p[2]*p[2]);}
    template<class Type> static inline float vnorm(const Type p[]) {return sqrt(vnorm2(p));}
    template<class Type> static inline void vnormalize(Type p[]) {Type n = vnorm2(p); if (n>0.0) vmultiply<Type>(p, (Type)1.0/sqrt(n), p);}
    template<class Type> static inline float vdistance2(const Type p[], const Type q[]) {return ((p[0]-q[0])*(p[0]-q[0]))+((p[1]-q[1])*(p[1]-q[1]))+((p[2]-q[2])*(p[2]-q[2]));}
    template<class Type> static inline float vdistance(const Type p[], const Type q[]) {return sqrt(vdistance2(p, q));}

    template<class Type> static inline void vorthogonal(const Type p[], Type q[])
    {
        Type qq[3];
        if (fabs(p[0])<=fabs(p[1]) && fabs(p[0])<=fabs(p[2]))
        {
            qq[0]=0.0f;qq[1]=p[2];qq[2]=-p[1];
        }
        else if (fabs(p[1])<=fabs(p[0]) && fabs(p[1])<=fabs(p[2]))
        {
            qq[0]=-p[2];qq[1]=0.;qq[2]=p[0];
        }
        else
        {
            qq[0]=p[1];qq[1]=-p[0];qq[2]=0.0f;
        }
        vcopy<Type>(qq,q);
        vnormalize<Type>(q);
    }

};

#endif // MM_VECTOR3_H
