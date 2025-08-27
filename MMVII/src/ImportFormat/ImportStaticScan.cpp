#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_ReadFileStruct.h"
#include "MMVII_util_tpl.h"
#include "MMVII_Geom3D.h"


/**
   \file importStaticScan.cpp

   \brief import static scan into instrument geometry
*/


namespace MMVII
{
/* ********************************************************** */
/*                                                            */
/*                 cAppli_ImportStaticScan                    */
/*                                                            */
/* ********************************************************** */

class cAppli_ImportStaticScan : public cMMVII_Appli
{
public :
    cAppli_ImportStaticScan(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
    int Exe() override;
    cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
    cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

    std::vector<std::string>  Samples() const override;
private :


    // Mandatory Arg
    std::string              mNameFile;
    std::string              mStationName;
    std::string              mScanName;

    // Optional Arg
    tREAL8                   mAngTolerancy;
    std::string              mTransfoIJK;

};

cAppli_ImportStaticScan::cAppli_ImportStaticScan(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli    (aVArgs,aSpec),
    mAngTolerancy   (1e-6),
    mTransfoIJK         ("ijk")
{
}

cCollecSpecArg2007 & cAppli_ImportStaticScan::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
           <<  Arg2007(mNameFile ,"Name of Input File",{eTA2007::FileAny})
           <<  Arg2007(mStationName ,"Station name",{eTA2007::Topo}) // TODO: change type to future station
           <<  Arg2007(mScanName ,"Scan name",{eTA2007::Topo}) // TODO: change type to future scan
        ;
}

cCollecSpecArg2007 & cAppli_ImportStaticScan::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return    anArgOpt
           << AOpt2007(mAngTolerancy,"AngTol","Angle tolerancy",{eTA2007::HDV})
           << AOpt2007(mTransfoIJK,"Transfo","Transfo to have primariy rotation axis as Z and X as theta origin",{{eTA2007::HDV}})
        ;
}

cPt3dr cart2spher(const cPt3dr & aPtCart) // returns theta phi dist
{
    // atan2($3,sqrt($1*$1+$2*$2))
    tREAL8 theta =  atan2(aPtCart.y(),aPtCart.x());
    tREAL8 dist = Norm2(aPtCart);
    tREAL8 distxy = sqrt(aPtCart.BigX2()+aPtCart.BigY2());
    tREAL8 phi =  atan2(aPtCart.z(),distxy);
    return {theta, phi, dist};
}

int cAppli_ImportStaticScan::Exe()
{
    cTriangulation3D<tREAL8> aTriangulation3DXYZ(mNameFile);

    StdOut() << "Got " <<aTriangulation3DXYZ.NbPts() <<" points.\n";
    if (aTriangulation3DXYZ.HasPtAttribute())
    {
        StdOut() << "Intensity found.\n";
    }

    StdOut() << "Sample:\n";
    for (size_t i=0; (i<10)&&(i<aTriangulation3DXYZ.VPts().size()); ++i)
    {
        StdOut() << aTriangulation3DXYZ.KthPts(i);
        if (aTriangulation3DXYZ.HasPtAttribute())
            StdOut() << " " << aTriangulation3DXYZ.KthPtsPtAttribute(i);
        StdOut() << "\n";
    }
    StdOut() << "...\n";

    cRotation3D<tREAL8>  aRotFrame = cRotation3D<tREAL8>::RotFromCanonicalAxes(mTransfoIJK);

    std::vector<cPt3dr> aVectPtsTPD(aTriangulation3DXYZ.NbPts()); // all points in theta-phi-dist
    for (size_t i=0; i<aVectPtsTPD.size(); ++i)
    {
        aVectPtsTPD[i] = cart2spher(aRotFrame.Value(aTriangulation3DXYZ.KthPts(i)));
    }

    // check theta-phi :
    StdOut() << "Spherical sample:\n";
    for (size_t i=0; (i<10)&&(i<aVectPtsTPD.size()); ++i)
    {
        StdOut() << aVectPtsTPD[i];
        StdOut() << "\n";
    }
    StdOut() << "...\n";

    cWhichMinMax<int, tREAL8> aMinMaxTheta;
    cWhichMinMax<int, tREAL8> aMinMaxPhi;
    for (const auto & aPtAng: aVectPtsTPD)
    {
        aMinMaxTheta.Add(0,aPtAng.x());
        aMinMaxPhi.Add(0,aPtAng.y());
    }
    cBox2dr aBoxAng( {aMinMaxTheta.Min().ValExtre(), aMinMaxPhi.Min().ValExtre()},
                     {aMinMaxTheta.Max().ValExtre(), aMinMaxPhi.Max().ValExtre()});
    StdOut() << "Box: " << aBoxAng << "\n";

    // find phi min and max diff
    // in absolute, diff min = step (and sign=direction), diff max = col heigth
    tREAL8 previousPhi = aVectPtsTPD.at(0).y();
    cWhichMinMax<size_t,tREAL8> aDiffPhi;
    for (const auto & aPtAng: aVectPtsTPD)
    {
        aDiffPhi.Add(0,aPtAng.y()-previousPhi);
    }
    StdOut() << "DiffPhi " << aDiffPhi.Min().ValExtre() <<  "   " << aDiffPhi.Max().ValExtre() << "\n";
    auto [phiStep,phiRange] = fabs(aDiffPhi.Min().ValExtre())<(aDiffPhi.Max().ValExtre()) ?
                                   std::make_pair(aDiffPhi.Min().ValExtre(), aDiffPhi.Max().ValExtre())
                                 : std::make_pair(aDiffPhi.Max().ValExtre(), aDiffPhi.Min().ValExtre());

    StdOut() << "phiStep " << phiStep <<  ",   " << fabs(phiRange)/fabs(phiStep)
             << " or " << (aBoxAng.P1().y()-aBoxAng.P0().y())/fabs(phiStep) << " steps\n";

    // find theta step
    tREAL8 aColChangeDetectorInPhistep = 10;
    tREAL8 previousTheta = aVectPtsTPD.at(0).x();
    previousPhi = aVectPtsTPD.at(0).y();
    std::vector<tREAL8> aVDiffColTheta;
    for (const auto & aPtAng: aVectPtsTPD)
    {
        StdOut() << "Pt "<<aPtAng<< " difphi " <<-(aPtAng.y()-previousPhi)/phiStep <<"\n";
        if (-(aPtAng.y()-previousPhi)/phiStep > aColChangeDetectorInPhistep)
        {
            aVDiffColTheta.push_back(aPtAng.x()-previousTheta);
        }
        previousTheta = aPtAng.x();
        previousPhi = aPtAng.y();
    }
    StdOut() << "DiffColTheta: ";
    for (auto & v: aVDiffColTheta)
        StdOut() << v <<" ";
    StdOut() << "\n";

    return EXIT_SUCCESS;
}

std::vector<std::string>  cAppli_ImportStaticScan::Samples() const
{
    return
        {

        };
}


tMMVII_UnikPApli Alloc_ImportStaticScan(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
    return tMMVII_UnikPApli(new cAppli_ImportStaticScan(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ImportStaticScan
    (
        "ImportStaticScan",
        Alloc_ImportStaticScan,
        "Import static scan cloud point into instrument raster geometry",
        {eApF::Cloud},
        {eApDT::Ply},
        {eApDT::MMVIICloud},
        __FILE__
        );

}; // MMVII

