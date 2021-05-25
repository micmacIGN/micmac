
#include "cRStats.h"

cRobustStats::cRobustStats(const Fonc_Num &aFonc,
                           const int &aNbV,
                           const Pt2di &aBrd,
                           const Pt2di &aSz) :
    mFonc(aFonc),
    mNbV(aNbV),
    mBrd(aBrd),
    mSz(aSz)
{
    DoCalc();
}

void cRobustStats::DoCalc()
{

    /* Analyses on data with outliers */

    //histogram on signed dh
    cHistoStats aHSh(mNbV,mFonc,mBrd,mSz,"StatsHistSign.txt");

    //histogram on absolute dh (for Q[0.68], Q[0.95])
    cHistoStats aHUSh(mNbV,Abs(mFonc),mBrd,mSz,"StatsHistAbs.txt");

    //standard accuracy measures
    double aSP, aMoy, aEc;
    CalcStdStat(mFonc,mBrd,mSz,aSP,aMoy,aEc,1);


    //NMAD, histogram
    double Q50 = aHSh.Quantile(round_ni(0.5*aSP));
    Fonc_Num aResMAD = Abs(mFonc- Q50);//NMAD is calculted on abs

    cHistoStats aHNMAD(mNbV,aResMAD,mBrd,mSz,"StatsHistMAD.txt");

    /* Analayses on data without outliers */
    int    aFoisSeuil=5;
    double aSeuil = double(aFoisSeuil)*aEc;

    //a trick to move from Fonc to Im2D_REAL4   
    std::string aNameTmp="tmp.tif";
    Tiff_Im::CreateFromFonc(aNameTmp,mSz,mFonc,GenIm::real4);
    Im2D_REAL4 aResIm = Im2D_REAL4::FromFileBasic(aNameTmp);

    //values below the Seil are retained, the rest is 0
    Im2D_REAL4 aResNOIm(mSz.x,mSz.y);
    ELISE_COPY
    (
        select(aResIm.all_pts(),Abs(aResIm.in())<aSeuil),
        aResIm.in(),
        aResNOIm.out()
    );
    Symb_FNum aResOutlier(aResIm.in()-aResNOIm.in());

    //standard accuracy measures
    double aSPNO, aMoyNO, aEcNO;
    ELISE_COPY//sum of non-outliers
    (
        rectangle(mBrd,mSz-mBrd),
        aResOutlier==0,
        sigma(aSPNO)
    );
    CalcStdStat(aResNOIm.in(),mBrd,mSz,aSPNO,aMoyNO,aEcNO,0);
    ELISE_fp::RmFile(aNameTmp);

    std::cout << "=== Accuracy measures by [Hoehle & Hoehle, 2009] ===\n";
    std::cout << "= Standard Accuracy measures \n";
    std::cout << aMoy << "    Mean\n";
    std::cout << aEc  << "    Std dev\n";
    std::cout << "                                                         \n";
    std::cout << aMoyNO << "    Mean (no outliers)\n";
    std::cout << aEcNO << "    Std dev (no outliers)\n";
    std::cout << aSeuil << "    Rejection threshold -> " << aFoisSeuil << "*std_dev or 50m\n";
    std::cout << aSP - aSPNO << "=" << (aSP-aSPNO)*100/aSP << "% Rejected outliers\n";


    std::cout << "= Robust Accuracy measures \n";
    std::cout << Q50 << "    Q(0.5) : median (50% quantile)\n";
    std::cout << 1.4826 * aHNMAD.Quantile(round_ni(0.5*aSP))  << "    NMAD\n";
    std::cout << aHUSh.Quantile(round_ni(0.683*aSP))  << "    Q(0.683) |dh| : 68% quantile\n";
    std::cout << aHUSh.Quantile(round_ni(0.95*aSP)) << "    Q(0.95) |dh| : 95% quantile\n";
    std::cout << aHUSh.Quantile(round_ni(1*aSP)) << "    Q(1.0)  |dh| :100% quantile\n";
    std::cout << "dh corresponding to a histogram bin for signed dh=" << 1/aHSh.Norm()  << "\n";
    std::cout << "dh corresponding to a histogram bin for abs dh=" << 1/aHUSh.Norm()  << "\n";

}

cHistoStats::cHistoStats(const int &aNbV,
                         const Fonc_Num &aFonc,
                         const Pt2di &aBrd,
                         const Pt2di &aSz,
                         const string & aName ) :
    mNbV (aNbV),
    mBrd (aBrd),
    mSz  (aSz),
    mName(aName),
    mMin (MinMax(aFonc,"MIN")),
    mMax (MinMax(aFonc,"MAX")),
    mNorm((mNbV-1)/(mMax-mMin)),
    mH   (InitH(aFonc))

{}

const Im1D_INT4 * cHistoStats::InitH(const Fonc_Num &aFonc)
{


    Im1D_INT4 *aH = new Im1D_INT4(mNbV,0);
    Flux_Pts aFlux = rectangle(mBrd,mSz-mBrd);

    ELISE_COPY
    (
        aFlux.chc(round_ni(Abs(aFonc-mMin)*mNorm)),
        1,
        aH->histo()
    );
    

    FILE * aFp = FopenNN(mName,"w","CmpIm");
    fprintf(aFp,"================= PERC  : RESIDU ==================\n");
    for (int aK=0 ; aK<mNbV ; aK++)
    {
            //std::cout << "aK " << aK << "=" << H.data()[aK] << "\n";
            fprintf(aFp,"Res[%f]=%d\n",aK/mNorm,aH->data()[aK]);
    }
    fclose(aFp);

    return(aH);
}

double cHistoStats::MinMax(const Fonc_Num &f, const string &MM)
{
    if(MM=="MAX")
    {
        double aRes;
        ELISE_COPY
        (
            rectangle(mBrd,mSz-mBrd),
            f,
            VMax(aRes)
        );
        return(aRes);

    }
    else if(MM=="MIN")
    {
        double aRes;
        ELISE_COPY
        (
            rectangle(mBrd,mSz-mBrd),
            f,
            VMin(aRes)
        );
        return(aRes);

    }
    else
        ELISE_ASSERT(false,"cHistoStats::MinMax => Either MIN or MAX");

    return(0);
}

double cHistoStats::Quantile(const int &aPr)
{

    int aCount=0;
    for(int aK=0; aK<mNbV; aK++)
    {
        aCount+=mH->data()[aK];

        if(aCount>=aPr)
            return(double(aK)/mNorm+mMin);
    }

    ELISE_ASSERT(false,"cHistoStats::Quantile  aPr overflows the available samples");
    return(0.0);
}

void cRobustStats::CalcStdStat(const Fonc_Num &aFonc,
                 const Pt2di &aBrd,
                 const Pt2di &aSz,
                 double &aSP,
                 double &aMoy,
                 double &aEc,
                 bool    OK)
{

    if(OK)//recalculates aSP
        ELISE_COPY
        (
            rectangle(aBrd,aSz-aBrd),
            1,
            sigma(aSP)
        );

    ELISE_COPY
    (
        rectangle(aBrd,aSz-aBrd),
        Virgule(aFonc,Square(aFonc)),
        Virgule
        (
             sigma(aMoy),
             sigma(aEc)
        )
    );
    aMoy /= aSP;
    aEc  /= aSP;
    aEc -= ElSquare(aMoy);
    aEc  = sqrt(ElMax(0.0,aEc));

}

