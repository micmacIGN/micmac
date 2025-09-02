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
    bool                     mNoMiss; // are every point present in cloud, even when no response?

    // data

};

cAppli_ImportStaticScan::cAppli_ImportStaticScan(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
    cMMVII_Appli    (aVArgs,aSpec),
    mAngTolerancy   (1e-6),
    mTransfoIJK     ("ijk"),
    mNoMiss         (false)
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

cPt3dr cart2spher(const cPt3dr & aPtCart)
{
    tREAL8 dist = Norm2(aPtCart);
    tREAL8 theta =  atan2(aPtCart.y(),aPtCart.x());
    tREAL8 distxy = sqrt(aPtCart.BigX2()+aPtCart.BigY2());
    tREAL8 phi =  atan2(aPtCart.z(),distxy);
    return {theta, phi, dist};
}

int cAppli_ImportStaticScan::Exe()
{
    tREAL8 aDistMinToExist = 0.01;
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
    size_t aNbPtsNul = 0;
    for (size_t i=0; i<aVectPtsTPD.size(); ++i)
    {
        aVectPtsTPD[i] = cart2spher(aRotFrame.Value(aTriangulation3DXYZ.KthPts(i)));
        if (aVectPtsTPD[i].z()<aDistMinToExist)
            aNbPtsNul++;
    }
    StdOut() << aNbPtsNul << " null points\n";

    // check theta-phi :
    StdOut() << "Spherical sample:\n";
    for (size_t i=0; (i<10)&&(i<aVectPtsTPD.size()); ++i)
    {
        StdOut() << aVectPtsTPD[i];
        StdOut() << "\n";
    }
    StdOut() << "...\n";

    mNoMiss = false;
    // et min max theta phi, check if there are points (0,0,0) <=> all points are present in cloud, even when no response
    cWhichMinMax<int, tREAL8> aMinMaxTheta;
    cWhichMinMax<int, tREAL8> aMinMaxPhi;
    for (const auto & aPtAng: aVectPtsTPD)
    {
        if (aPtAng.z()<aDistMinToExist)
        {
            mNoMiss = true;
            continue;
        }
        aMinMaxTheta.Add(0,aPtAng.x());
        aMinMaxPhi.Add(0,aPtAng.y());
    }
    cBox2dr aBoxAng( {aMinMaxTheta.Min().ValExtre(), aMinMaxPhi.Min().ValExtre()},
                     {aMinMaxTheta.Max().ValExtre(), aMinMaxPhi.Max().ValExtre()});
    StdOut() << "Box: " << aBoxAng << "\n";

    // find phi min diff
    // successive phi diff is useful, but only if we are in the same scanline
    tREAL8 previousTheta = NAN;
    tREAL8 previousPhi = NAN;
    tREAL8 phiStep = INFINITY; // signed value that is min in abs. For successive points on one column
    tREAL8 angularPrecisionInSteps = 1; // we suppose that theta changes slower than phi... Is it ok???
    for (const auto & aPtAng: aVectPtsTPD)
    {
        if (aPtAng.z()<aDistMinToExist) continue;
        auto aDiffPhi = aPtAng.y()-previousPhi;
        auto aDiffTheta = aPtAng.x()-previousTheta;
        if (fabs(aDiffTheta)<fabs(phiStep)*angularPrecisionInSteps) // we are on the same scanline
        {
            if (fabs(aDiffPhi)<fabs(phiStep))
            {
                //std::cout<<"with prev "<<previousTheta<< " "<< previousPhi<< "  curr "<<aPtAng<<":\n";
                //std::cout<<"up: "<<minDiffPhi <<" " <<aDiffPhi<<"\n";
                phiStep = aDiffPhi;
            }
        }
        previousTheta = aPtAng.x();
        previousPhi = aPtAng.y();
    }
    StdOut() << "phiStep " << phiStep << ",  " << (aBoxAng.P1().y()-aBoxAng.P0().y())/fabs(phiStep) << " steps\n";

    tREAL8 aColChangeDetectorInPhistep = 100;
    /*
    // find theta step
    previousTheta = NAN;
    previousPhi = NAN;
    std::vector<tREAL8> aVDiffColTheta;
    for (const auto & aPtAng: aVectPtsTPD)
    {
        if (aPtAng.z()<aDistMinToExist) continue;
        //StdOut() << "Pt "<<aPtAng<< " difphi " <<-(aPtAng.y()-previousPhi)/phiStep <<"\n";
        if (-(aPtAng.y()-previousPhi)/phiStep > aColChangeDetectorInPhistep)
        {
            aVDiffColTheta.push_back(aPtAng.x()-previousTheta);
        }
        previousTheta = aPtAng.x();
        previousPhi = aPtAng.y();
    }
    StdOut() << "Found "<<aVDiffColTheta.size()<<" cols\n";
    //StdOut() << "DiffColTheta: ";
    //for (auto & v: aVDiffColTheta)
    //    StdOut() << v <<" ";
    //StdOut() << "\n";
    */

    // compute line and col for each point
    std::vector<int> aVectPtsLine(aTriangulation3DXYZ.NbPts());
    std::vector<int> aVectPtsCol(aTriangulation3DXYZ.NbPts());
    int aMaxCol = 0;
    previousPhi = NAN;
    previousTheta = NAN;
    int aMaxLine = 0;
    int aCurrLine = 0;
    for (size_t i=0; i<aVectPtsTPD.size(); ++i)
    {
        //if (aMaxCol==12191/2)
        //    std::cout<<"  "<<i<<" "<<aCurrLine<<"\n";
        auto aPtAng = aVectPtsTPD[i];
        if (aPtAng.z()<aDistMinToExist)
        {
            aVectPtsLine[i] = 0;
            aVectPtsCol[i] = 0;
            aCurrLine++;
            continue;
        }
        if (mNoMiss)
            aCurrLine++;
        else
            aCurrLine = (aPtAng.y()-aBoxAng.P0().y())/fabs(phiStep);

        if (aCurrLine>aMaxLine) aMaxLine = aCurrLine;

        if (-(aPtAng.y()-previousPhi)/phiStep > aColChangeDetectorInPhistep)
        {
            aMaxCol++;
            //StdOut() << "change col: "<<previousTheta<<" "<<previousPhi<<"  -  "<<aPtAng.x()<<" "<<aPtAng.y()<<"\n";
            aCurrLine=0;
        }
        aVectPtsLine[i] = aCurrLine;
        aVectPtsCol[i] = aMaxCol;
        previousTheta = aPtAng.x();
        previousPhi = aPtAng.y();
    }
    StdOut() << "Max col found: "<<aMaxCol<<"\n";
    StdOut() << "Max line found: "<<aMaxLine<<"\n";

    StdOut() << "Image size: "<<cPt2di(aMaxCol+1, aMaxLine+1)<<"\n";
    //fill rasters
    cIm2D<tU_INT1> aRasterIntens(cPt2di(aMaxCol+1, aMaxLine+1), 0, eModeInitImage::eMIA_Null);
    auto & aRasterIntensData = aRasterIntens.DIm();
    cIm2D<tU_INT4> aRasterIndex(cPt2di(aMaxCol+1, aMaxLine+1), 0, eModeInitImage::eMIA_Null);
    auto & aRasterIndexData = aRasterIndex.DIm();
    cIm2D<float> aRasterTheta(cPt2di(aMaxCol+1, aMaxLine+1), 0, eModeInitImage::eMIA_Null);
    auto & aRasterThetaData = aRasterTheta.DIm();
    cIm2D<float> aRasterPhi(cPt2di(aMaxCol+1, aMaxLine+1), 0, eModeInitImage::eMIA_Null);
    auto & aRasterPhiData = aRasterPhi.DIm();
    cIm2D<float> aRasterDist(cPt2di(aMaxCol+1, aMaxLine+1), 0, eModeInitImage::eMIA_Null);
    auto & aRasterDistData = aRasterDist.DIm();
    cIm2D<tU_INT1> aRasterMask(cPt2di(aMaxCol+1, aMaxLine+1), 0, eModeInitImage::eMIA_Null);
    auto & aRasterMasksData = aRasterMask.DIm();
    for (size_t i=0; i<aVectPtsTPD.size(); ++i)
    {
        auto aPtAng = aVectPtsTPD[i];
        cPt2di aPcl = {aVectPtsCol[i], aMaxLine-aVectPtsLine[i]};
        //if (aVectPtsCol[i]==12191/2)
        //    std::cout<<"  "<<i<<" "<<aPcl<<"\n";
        if (aPtAng.z()<aDistMinToExist)
        {
            aRasterMasksData.SetV(aPcl, 1);
            continue;
        }
        aRasterIntensData.SetV(aPcl, aTriangulation3DXYZ.KthPtsPtAttribute(i)*255);
        aRasterIndexData.SetV(aPcl, i);
        aRasterThetaData.SetV(aPcl, aPtAng.x());
        aRasterPhiData.SetV(aPcl, aPtAng.y());
        aRasterDistData.SetV(aPcl, aPtAng.z());
    }
    aRasterIntensData.ToFile("totoIntens.png");
    aRasterIndexData.ToFile("totoIndex.tif");
    aRasterThetaData.ToFile("totoTheta.tif");
    aRasterPhiData.ToFile("totoPhi.tif");
    aRasterDistData.ToFile("totoDist.tif");
    aRasterMasksData.ToFile("totoMask.png");



    // estimate verticalization correction if scanner with compensator
    int aNbPlanes = 10; // try to get several planes for instrument primariy axis estimation
    float aCorrectPlanePhiRange = 80*M_PI/180; // try to get points with this phi diff in a scanline
    int aColPlaneStep = aMaxCol / aNbPlanes;
    int aLineGoodRange = aCorrectPlanePhiRange/fabs(phiStep);
    if (aLineGoodRange > aMaxLine - 2)
        aLineGoodRange = aMaxLine - 2; // for small scans, use full height

    int aTargetCol = 0; // the next we search for
    int aTargetLine = 0;
    previousPhi = NAN;
    previousTheta = NAN;
    int aCurrCol = 0;
    aCurrLine = 0;
    std::vector<std::tuple<cPt3dr, cPt3dr, cPt3dr>> aVPtsPlanes; // list of triplets to find vertical planes
    cPt3dr * aPtBottom = nullptr;
    cPt3dr * aPtTop = nullptr;
    for (size_t i=0; i<aVectPtsTPD.size(); ++i)
    {
        // TODO: factorize xyz points list to linecol!
        auto aPtAng = aVectPtsTPD[i];
        if (aPtAng.z()<aDistMinToExist)
        {
            aCurrLine++;
            continue;
        }
        if (mNoMiss)
            aCurrLine++;
        else
            aCurrLine = (aPtAng.y()-aBoxAng.P0().y())/fabs(phiStep);

        if (-(aPtAng.y()-previousPhi)/phiStep > aColChangeDetectorInPhistep)
        {
            aCurrCol++;
            aCurrLine=0;
        }
        previousTheta = aPtAng.x();
        previousPhi = aPtAng.y();
        if (aCurrCol == aTargetCol)
        {
            if (!aPtBottom)
            {
                aPtBottom = &aTriangulation3DXYZ.KthPts(i);
                aTargetLine = aCurrLine + aLineGoodRange;
            }
            else if ((aCurrLine>aTargetLine)&&(!aPtTop))
            {
                aPtTop = &aTriangulation3DXYZ.KthPts(i);
                aVPtsPlanes.push_back( {cPt3dr(0.,0.,0.),(*aPtBottom)/Norm2(*aPtBottom), (*aPtTop)/Norm2(*aPtTop)} );
                aPtBottom = nullptr;
                aPtTop = nullptr;
                aTargetCol = aCurrCol + aColPlaneStep;
            }
        }
    }

    std::vector<cPlane3D> aVPlanes;
    for (auto & [aP0,aP1,aP2] :aVPtsPlanes)
    {
        aVPlanes.push_back(cPlane3D::From3Point(aP0,aP1,aP2));
    }
    tSeg3dr aSegVert = cPlane3D::InterPlane(aVPlanes, aNbPlanes/2);
    StdOut() << "Vert: " << aSegVert.V12() << " " << aSegVert.P2() << "\n";

    cRotation3D<tREAL8> aVertRot = cRotation3D<tREAL8>::CompleteRON(aSegVert.V12(),2);

    // update xyz and tpd coordinates
    for (size_t i=0; i<aTriangulation3DXYZ.NbPts(); ++i)
    {
        aTriangulation3DXYZ.KthPts(i) = aVertRot.Inverse(aTriangulation3DXYZ.KthPts(i));
        aVectPtsTPD[i] = cart2spher(aRotFrame.Value(aTriangulation3DXYZ.KthPts(i)));
    }


    // make statistics on theta phi, using raster geometry
    StdOut() << "Compute steps from inital raster geometry\n";
    //std::vector<tREAL8> aVDiffTheta;
    std::fstream file_theta;
    file_theta.open("thetas.txt", std::ios_base::out);
    std::fstream file_theta_abs;
    file_theta_abs.open("thetas_abs.txt", std::ios_base::out);
    tREAL8 aThetaStep2 = 0.;
    int nbThetaStep = 0;
    for (int c=1+aMaxCol/3; c<2*aMaxCol/3+1; ++c)
    {
        tU_INT4 i = aRasterIndexData.GetV(cPt2di(c,aMaxLine/2));
        tU_INT4 ic = aRasterIndexData.GetV(cPt2di(c-1,aMaxLine/2));
        file_theta_abs<<i<<" "<<aVectPtsTPD[i].x()<< " "<<ic<<" "<<aVectPtsTPD[ic].x()<<"\n";
        if ((i!=0)&&(ic!=0))
        {
            if ((aVectPtsTPD[i].z()<aDistMinToExist)
                ||(aVectPtsTPD[ic].z()<aDistMinToExist))
            {
                std::cout<<"EEEERRRROROORORORO\n";
                std::cout<<i<<" "<<aVectPtsTPD[i]<< " "<<ic<<" "<<aVectPtsTPD[ic]<<"\n";
            }
            auto aDiffTheta = aVectPtsTPD[i].x()-aVectPtsTPD[ic].x();
            if (aDiffTheta>M_PI)
                aDiffTheta -= 2*M_PI;
            if (aDiffTheta<-M_PI)
                aDiffTheta += 2*M_PI;
            file_theta<<aDiffTheta<<"\n";
            aThetaStep2+=aDiffTheta;
            nbThetaStep++;
        }
    }
    file_theta.close();
    file_theta_abs.close();
    aThetaStep2/=nbThetaStep;
    StdOut() << " New Theta Step: " << aThetaStep2<<"\n";

    std::fstream file_phi;
    file_phi.open("phis.txt", std::ios_base::out);
    std::fstream file_phi_abs;
    file_phi_abs.open("phis_abs.txt", std::ios_base::out);
    tREAL8 aPhiStep2 = 0.;
    int nbPhiStep = 0;
    for (int l=1+aMaxLine/3; l<2*aMaxLine/3+1; ++l)
    {
        tU_INT4 i = aRasterIndexData.GetV(cPt2di(aMaxCol/2,l));
        tU_INT4 il = aRasterIndexData.GetV(cPt2di(aMaxCol/2,l-1));
        file_phi_abs<<i<<" "<<aVectPtsTPD[i].y()<< " "<<il<<" "<<aVectPtsTPD[il].y()<<"\n";
        if ((i!=0)&&(il!=0))
        {
            if ((aVectPtsTPD[i].z()<aDistMinToExist)
                ||(aVectPtsTPD[il].z()<aDistMinToExist))
            {
                std::cout<<"EEEERRRROROORORORO\n";
                std::cout<<i<<" "<<aVectPtsTPD[i]<< " "<<il<<" "<<aVectPtsTPD[il]<<"\n";
            }
            file_phi<<(aVectPtsTPD[i].y()-aVectPtsTPD[il].y())<<"\n";
            aPhiStep2+=(aVectPtsTPD[i].y()-aVectPtsTPD[il].y());
            nbPhiStep++;
        }
    }
    file_phi.close();
    file_phi_abs.close();
    aPhiStep2/=nbPhiStep;
    StdOut() << " New Phi Step: " << aPhiStep2<<"\n";

    /*
    float minDiffPhi = INFINITY;
    float minDiffTheta = INFINITY;
    int signDiffPhi = 0;
    int signDiffTheta = 0;
    for (int l=1; l<aMaxLine+1; ++l)
    {
        for (int c=1; c<aMaxCol+1; ++c)
        {
            tU_INT4 i = aRasterIndexData.GetV(cPt2di(c,l));
            tU_INT4 ic = aRasterIndexData.GetV(cPt2di(c-1,l));
            tU_INT4 il = aRasterIndexData.GetV(cPt2di(c,l-1));
            if ((i!=0)&&(ic!=0)&&(il!=0))
            {
                if ((aVectPtsTPD[i].z()<aDistMinToExist)
                    ||(aVectPtsTPD[ic].z()<aDistMinToExist)
                    ||(aVectPtsTPD[il].z()<aDistMinToExist))
                {
                    std::cout<<"EEEERRRROROORORORO\n";
                    std::cout<<i<<" "<<aVectPtsTPD[i]<< " "<<ic<<" "<<aVectPtsTPD[ic]<<" "<<il<<" "<<aVectPtsTPD[il]<<"\n";
                }
                float aDiffTheta = aVectPtsTPD[i].x()-aVectPtsTPD[ic].x();
                if (minDiffTheta>fabs(aDiffTheta))
                {
                    std::cout<<"up th:"<<aDiffTheta<<"\n";
                    minDiffTheta = fabs(aDiffTheta);
                    signDiffTheta = fabs(aDiffTheta)/aDiffTheta;
                }
                float aDiffPhi = aVectPtsTPD[i].y()-aVectPtsTPD[il].y();
                if (minDiffPhi>fabs(aDiffPhi))
                {
                    std::cout<<"up ph:"<<aDiffTheta<<"\n";
                    minDiffPhi = fabs(aDiffPhi);
                    signDiffPhi = fabs(aDiffPhi)/aDiffPhi;
                }
                //aVDiffTheta.push_back(aVectPtsTPD[i2].x()-aVectPtsTPD[i1].x());
            }
        }
    }
    StdOut() << "Phi step found: "<<signDiffPhi*minDiffPhi<<"\n";
    StdOut() << "Theta step found: "<<signDiffTheta*minDiffTheta<<"\n";
    */

    // save pictures using steps
    aMaxLine = 0;
    aMaxCol = 0;
    aCurrLine = 0;
    aCurrCol = 0;
    for (size_t i=0; i<aVectPtsTPD.size(); ++i)
    {
        auto aPtAng = aVectPtsTPD[i];
        if (aPtAng.z()<aDistMinToExist)
        {
            aVectPtsLine[i] = 0;
            aVectPtsCol[i] = 0;
            continue;
        }
        if (aPhiStep2>0)
            aCurrLine = (aPtAng.y()-aBoxAng.P0().y())/aPhiStep2;
        else
            aCurrLine = (aPtAng.y()-aBoxAng.P1().y())/aPhiStep2;
        if (aCurrLine>aMaxLine) aMaxLine = aCurrLine;

        if (aThetaStep2>0)
            aCurrCol = (aPtAng.x()-aBoxAng.P0().x())/aThetaStep2;
        else
            aCurrCol = (aPtAng.x()-aBoxAng.P1().x())/aThetaStep2;
        if (aCurrCol>aMaxCol) aMaxCol = aCurrCol;

        aVectPtsLine[i] = aCurrLine;
        aVectPtsCol[i] = aCurrCol;
    }
    StdOut() << "Max col found: "<<aMaxCol<<"\n";
    StdOut() << "Max line found: "<<aMaxLine<<"\n";

    StdOut() << "Image2 size: "<<cPt2di(aMaxCol+1, aMaxLine+1)<<"\n";
    //fill rasters
    cIm2D<tU_INT1> aRasterIntens2(cPt2di(aMaxCol+1, aMaxLine+1), 0, eModeInitImage::eMIA_Null);
    auto & aRasterIntens2Data = aRasterIntens2.DIm();
    for (size_t i=0; i<aVectPtsTPD.size(); ++i)
    {
        auto aPtAng = aVectPtsTPD[i];
        cPt2di aPcl = {aVectPtsCol[i], aVectPtsLine[i]};
        //if (aVectPtsCol[i]==12191/2)
        //    std::cout<<"  "<<i<<" "<<aPcl<<"\n";
        if (aPtAng.z()<aDistMinToExist)
        {
            continue;
        }
        aRasterIntens2Data.SetV(aPcl, aTriangulation3DXYZ.KthPtsPtAttribute(i)*255);
    }
    aRasterIntens2Data.ToFile("titiIntens.png");


    // search for min/max line and col with step
    float aMinLinef = INFINITY;
    float aMinColf = INFINITY;
    float aMaxLinef = -INFINITY;
    float aMaxColf = -INFINITY;
    float aCurrLineFloat, aCurrColFloat;
    for (size_t i=0; i<aVectPtsTPD.size(); ++i)
    {
        auto aPtAng = aVectPtsTPD[i];
        if (aPtAng.z()<aDistMinToExist)
        {
            continue;
        }
        if (aPhiStep2>0)
            aCurrLineFloat = (aPtAng.y()-aBoxAng.P0().y())/aPhiStep2;
        else
            aCurrLineFloat = (aPtAng.y()-aBoxAng.P1().y())/aPhiStep2;
        if (aThetaStep2>0)
            aCurrColFloat = (aPtAng.x()-aBoxAng.P0().x())/aThetaStep2;
        else
            aCurrColFloat = (aPtAng.x()-aBoxAng.P1().x())/aThetaStep2;
        if (aMinLinef > aCurrLineFloat)
            aMinLinef = aCurrLineFloat;
        if (aMinColf > aCurrColFloat)
            aMinColf = aCurrColFloat;
        if (aMaxLinef < aCurrLineFloat)
            aMaxLinef = aCurrLineFloat;
        if (aMaxColf < aCurrColFloat)
            aMaxColf = aCurrColFloat;
    }
    StdOut() << " " << aMinColf << "  " << aMaxColf << " " << aMinLinef << "  " << aMaxLinef << std::endl;
    cIm2D<float> aRasterDenity(cPt2di(aMaxColf-aMinColf+3, aMaxLinef-aMinLinef+3), 0, eModeInitImage::eMIA_Null); // +3 pixels for safety
    auto & aRasterDensityData = aRasterDenity.DIm();
    StdOut() << "aRasterDensityData.Sz(): " << aRasterDensityData.Sz() <<"\n";
    for (size_t i=0; i<aVectPtsTPD.size(); ++i)
    {
        auto aPtAng = aVectPtsTPD[i];
        if (aPtAng.z()<aDistMinToExist)
        {
            continue;
        }
        if (aPhiStep2>0)
            aCurrLineFloat = (aPtAng.y()-aBoxAng.P0().y())/aPhiStep2;
        else
            aCurrLineFloat = (aPtAng.y()-aBoxAng.P1().y())/aPhiStep2;
        if (aThetaStep2>0)
            aCurrColFloat = (aPtAng.x()-aBoxAng.P0().x())/aThetaStep2;
        else
            aCurrColFloat = (aPtAng.x()-aBoxAng.P1().x())/aThetaStep2;
        //StdOut()<<aCurrColFloat+2 << " " << aCurrLineFloat+2 <<"\n";
        auto aP2d = cPt2dr(aCurrColFloat+aMinColf+1., aCurrLineFloat+aMinLinef+1.);
        if (!aRasterDensityData.InsideBL(aP2d))
        {
            StdOut()<<"pb: " << aP2d <<"\n";
        }
        aRasterDensityData.AddVBL(aP2d, 1.);
    }
    aRasterDensityData.ToFile("titiIntens.tif");


/*cIm2D<float> aRasterOrder(cPt2di(aMaxCol+1, aMaxLine+1), 0, eModeInitImage::eMIA_Null);
    auto & aRasterOrderData = aRasterOrder.DIm();
    for (size_t i=0; (i<10000) && (i<aVectPtsTPD.size()); ++i)
    {
        auto aPtAng = aVectPtsTPD[i];
        if (aPtAng.z()<aDistMinToExist) continue;
        aRasterOrderData.SetV(cPt2di(aMaxCol-aVectPtsCol[i], aMaxLine-aVectPtsLine[i]), i);
    }
    aRasterOrderData.ToFile("order.tif");*/

    // export clouds for debug
    #include <fstream>
    std::fstream file1;
    file1.open("cloud.xyz", std::ios_base::out);
    std::fstream file2;
    file2.open("cloud_norm.xyz", std::ios_base::out);
    for (size_t i=0; i<aTriangulation3DXYZ.VPts().size(); ++i)
    {
        if ((i/aMaxLine) % (aMaxCol/12) == 0)
        {
            int r = 127 + 127 * sin(i/1000. + 0*M_PI/3);
            int g = 127 + 127 * sin(i/1000. + 1*M_PI/3);
            int b = 127 + 127 * sin(i/1000. + 2*M_PI/3);

            //auto aPtAng = aVectPtsTPD[i];
            //StdOut() << " "<<aPtAng.x()<<" "<<aPtAng.y()<<"\n";
            auto aPt = aVertRot.Inverse(aTriangulation3DXYZ.KthPts(i));
            auto norm = Norm2(aPt);
            file1 << aPt.x() << " " << aPt.y() << " " << aPt.z() << " " << r << " " << g << " " << b << "\n"; //i << " " << aTriangulation3DXYZ.KthPtsPtAttribute(i) << "\n";
            file2 << aPt.x()/norm << " " << aPt.y()/norm << " " << aPt.z()/norm << " " << r << " " << g << " " << b << "\n"; //<< i << " " << aTriangulation3DXYZ.KthPtsPtAttribute(i) << "\n";
        }
    }
    file2.close();
    file1.close();

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

