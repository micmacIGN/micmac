#include "MMVII_Ptxd.h"
#include "MMVII_SysSurR.h"

#include <vector>

/*
double f(double x, double y)
{
    return 10 * sin(x) + sin(y);
}

cPolyXY_N<double> P(mDegree);

for (double x = -3.0; x<=3.0; x+= 0.1) {
    for (double y = -3.0; y<=3.0; y+= 0.1) {
        P.AddObs(x,y, f(x,y));
    }
}
P.Fit();
printf("P var %lf\n",P.VarCurSol());
for (int i=0; i< 10; i++) {
    cPt2dr p(RandInInterval(-3,3),RandInInterval(-3,3));
    auto v = f(p.x(),p.y());
    auto po = P(p);
    StdOut() << std::setw(11) << po << " " << std::setw(11) << v << " " << std::setw(11) << v - po  << " " << p << "\n";
}
*/

namespace MMVII
{
template<typename T>
class cPolyXY_N
{
public:
    explicit cPolyXY_N(int degree);
    explicit cPolyXY_N(std::initializer_list<T> aVK);
    explicit cPolyXY_N(const std::vector<T>& aVK);

    T operator()(const T& x, const T& y);
    T operator()(const cPtxd<T,2>& aPt);

    const T& K(int i, int j) const;
    T& K(int i, int j);
    const T& K(int n) const;
    T& K(int n);
    const std::vector<T>& VK() const;
    template <typename IT>
    void SetVK(IT it);

    int Degree() const;
    int NbCoeffs() const;

    template <typename IT>
    T VarToCoeffs(const T &x, const T &y, IT CoeffIt, const T& aFactor);
    template <typename IT>
    T VarToCoeffs(const cPtxd<T,2>& aPt, IT CoeffIt, const T& aFactor);

    void ResetFit();
    void AddFixedK(int i, int j, const T& k);
    void AddObs(const T& x, const T& y, const T& v, const T& aWeight=1);
    void AddObs(const cPtxd<T,2> aPt, const T& v, const T& aWeight=1);
    void Fit();
    T VarCurSol() const;


private:
    int idx(int i, int j) const;
    int mDegree;
    std::vector<T> mVK;
    std::unique_ptr<cLeasSqtAA<T>> mLeastSq;
    std::map<std::pair<int,int>, T> mFixedK;
};

} // MMVII

