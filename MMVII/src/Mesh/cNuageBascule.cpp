#include "cMMVII_Appli.h"
#include "MMVII_PCSens.h"
#include "MMVII_Geom2D.h"
#include "MMVII_Geom3D.h"
#include "MMVII_AllClassDeclare.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_Triangles.h"
#include "MMVII_Image2D.h"
#include "MMVII_ZBuffer.h"
#include "MeshDev.h"
#include "MMVII_Sys.h"
#include "MMVII_Radiom.h"
#include <fstream>
#include <iostream>
#include <StdAfx.h>



namespace MMVII
{
 class cAppliNuageBascule : public cMMVII_Appli,
                            public cAppliParseBoxIm<tREAL4>
{
    private:
        int Exe() override;
        int ExeOnParsedBox() override;


        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        std::string mNameCloud2D_DepthIn;
        //std::string mNameIm;
        std::string mNameMasq1;
        std::string mNameCorrel;
        std::string mNameSec;

        cSensorCamPC * mCamPC;
        cSensorCamPC * mSecCamPC;
        cIm2D<tREAL4> mImPx1;
        cIm2D<tU_INT1> mImMasq1;
        cIm2D<tU_INT1> mImCorrel;
        cIm2D<tU_INT1> mImMasqOut;
        cIm2D<tU_INT1> mImCorrelOut;
        cIm2D<tREAL4> mImRed;

        // photogrammetric project
        cPhotogrammetricProject mPhProj;
        cTriangulation3D<tREAL8> * mTri3D;
        std::string mNameResult;
        eModeGeom mModeGeom;
        tREAL8 mGSD;
        std::string GSD;
        tREAL8 mNoiseZ;
        tREAL8 mThreshGrad;
        bool mZF_SameOri;
        bool mBascCorrel;
        int  mMultZ;
        double      mMII;   ///<  Marge Inside Image
        //cBox2dr mBoxLocTarget;
        cTplBoxOfPts<tREAL8,2> mBoxLocTarget;
        cTplBoxOfPts<tREAL8,2> mBoxGlobTarget;
        std::vector<tU_INT1> mVCorrel;
        std::vector <tU_INT1> mVMasq;

    public:
        cAppliNuageBascule(const std::vector<std::string> & aVArgs, const cSpecMMVII_Appli & aSpec );
        typedef cAppliParseBoxIm<tREAL4> tAPBI;
        typedef tAPBI::tIm               tImAPBI;
        typedef tAPBI::tDataIm           tDImAPBI;
        static constexpr tREAL8 mInfty =  -1e10;
        cPt3dr BascOnePoint(cPt2di A,  cPt2di anOffSet);
        void MakeBasc();
        void MakeBasculeTris(cZBuffer & aZB);
        void ProcessNoPix(cZBuffer &  aZB);
        void AnalyzeSurfParams();
        void GenTFW(const cAffin2D<tREAL8> & anAff, const std::string & aNameTFW);
        cAffin2D<tREAL8> ReadTFW(const std::string & aNameTFW);
        cBox2di  BoxUtile();
        void MergeResults();
};


cAppliNuageBascule::cAppliNuageBascule(const std::vector<std::string> & aVArgs, const cSpecMMVII_Appli & aSpec ):
    cMMVII_Appli(aVArgs,aSpec),
    cAppliParseBoxIm<tREAL4>(*this,eForceGray::No,cPt2di(2000,2000),cPt2di(50,50),true),
    mCamPC(nullptr),
    mSecCamPC(nullptr),
    mImPx1(cPt2di(1,1)),
    mImMasq1(cPt2di(1,1)),
    mImCorrel(cPt2di(1,1)),
    mImMasqOut(cPt2di(1,1)),
    mImCorrelOut(cPt2di(1,1)),
    mImRed(cPt2di(1,1)),
    mPhProj(*this),
    mTri3D(nullptr),
    mNameResult("Bascule"),
    mModeGeom(eModeGeom::eGEOM_EPIP),
    mGSD(0.2),
    mNoiseZ(0.2),
    mThreshGrad(0.3),
    mZF_SameOri(true),
    mBascCorrel(false),
    mMultZ(mZF_SameOri ? 1 : -1),
    mMII(0.0)
{
}


cCollecSpecArg2007 & cAppliNuageBascule::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return
            APBI_ArgObl(anArgObl)
           <<   Arg2007(mNameCloud2D_DepthIn,"Name of input depth map", {eTA2007::FileImage} )
           <<   mPhProj.DPOrient().ArgDirInMand()
           <<   mPhProj.DPMeshDev().ArgDirOutMand()
        ;
}


cCollecSpecArg2007 & cAppliNuageBascule::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return
        APBI_ArgOpt
        (
            anArgOpt
                << AOpt2007(mModeGeom,"ModeGeom","Either epipolar geometry, def=geometry of input depth",{AC_ListVal<eModeGeom>()})
                << AOpt2007(mNameResult,"Out"," prefix of output files, default=Bascule.tif",{eTA2007::HDV})
                << AOpt2007(mNameMasq1, "Masq1","Masq of first image if any")
                << AOpt2007(mNameSec, "Im2","Secondary image if geom is Epip")
                << AOpt2007(mNameCorrel,"ImCorrel","Name of correlation or confidence image")
                << AOpt2007(GSD,"GroundResolution", "Ground sampling distance of bascule")
                << AOpt2007(mMII,"MII","Margin Inside Image (for triangle validation)", {eTA2007::HDV})
               /*<< AOpt2007(mResolZBuf,"ResZBuf","Resolution of ZBuffer", {eTA2007::HDV})
               << AOpt2007(mDoImages,"DoIm","Do images", {eTA2007::HDV})
               << AOpt2007(mNbPixImRedr,"NbPixIR","Resolution of ZBuffer", {eTA2007::HDV})
               << AOpt2007(mMII,"MII","Margin Inside Image (for triangle validation)", {eTA2007::HDV})
               << AOpt2007(mSKE,CurOP_SkipWhenExist,"Skip command when result exist")*/
               //<< AOptBench()
               //<< mPhProj.DPRadiomData().ArgDirOutOpt()
        )
        ;
}

cBox2di cAppliNuageBascule::BoxUtile()
{
    cTplBoxOfPts<int,2> aBox;
    for (const auto & aPix: mImMasq1.DIm())
    {
        if(mImMasq1.DIm().GetV(aPix)>0.99)
            aBox.Add(aPix);
    }

    return aBox.CurBox();
}


cPt3dr cAppliNuageBascule::BascOnePoint(cPt2di A,  cPt2di anOffSet)
{
    cPt2di  aPix1 = A+anOffSet;
    double  aX2= aPix1.x() + mImPx1.DIm().GetV(A);
    cPt2dr aPix2=cPt2dr(aX2, aPix1.y());
    tSeg3dr aP2TerCam1 = mCamPC->Image2Bundle(ToR(aPix1));
    tSeg3dr aP2TerCam2= mSecCamPC->Image2Bundle(aPix2);
    cPt3dr aPZA=RobustBundleInters({aP2TerCam1,aP2TerCam2});
    mBoxLocTarget.Add(Proj(aPZA));
    return aPZA;
}


void cAppliNuageBascule::MakeBasc()
{
    //mImRed.DIm().Resize(CurSzIn(),eModeInitImage::eMIA_Null);
    cPt2di aP0 = CurP0();
    cPt2di aPix1;
    std::vector<cPt3dr> aVPts;
    std::vector<cPt3di> aVFaces;

    cBox2di  aBoxUtileWithMasq = BoxUtile();

    ///  Init index for faces
    ///
    cIm2D<int> anImIndex(aBoxUtileWithMasq.Sz());
    cDataIm2D<int> & aDIdx= anImIndex.DIm();

    int IndBegin=0;
    for (const auto & aPix: aDIdx)
    {
        aDIdx.SetV(aPix,IndBegin);
        IndBegin++;
    }


    /// Create mesh points
    for(const auto & aPix: cPixBox<2>(aBoxUtileWithMasq.P0(),
                                       aBoxUtileWithMasq.P1())
         )
    {
        cPt2di P00(aPix.x(),aPix.y());
        cPt3dr aP00_3D =BascOnePoint(P00,aP0);
        aVPts.push_back(aP00_3D);

        if(mBascCorrel)
            mVCorrel.push_back(mImCorrel.DIm().GetV(aPix));
    }


    /// Create mesh faces
    for(const auto & aPix: cPixBox<2>(aBoxUtileWithMasq.P0(),
                                       aBoxUtileWithMasq.P1()-cPt2di(1,1))
         )
    {
        cPt2di P00(aPix.x(),aPix.y());
        cPt2di P10(aPix.x()+1,aPix.y());
        cPt2di P01(aPix.x(),aPix.y()+1);
        cPt2di P11(aPix.x()+1,aPix.y()+1);

        cPt3di aFUP(aDIdx.GetV(P00-aBoxUtileWithMasq.P0()),
                    aDIdx.GetV(P10-aBoxUtileWithMasq.P0()),
                    aDIdx.GetV(P11-aBoxUtileWithMasq.P0()));
        cPt3di aFDOWN(aDIdx.GetV(P00-aBoxUtileWithMasq.P0()),
                      aDIdx.GetV(P11-aBoxUtileWithMasq.P0()),
                      aDIdx.GetV(P01-aBoxUtileWithMasq.P0()));

        aVFaces.push_back(aFUP);
        aVFaces.push_back(aFDOWN);
    }
    StdOut()<<"computed tri "<<std::endl;


    //APBI_WriteIm(mNameResult,mImRed);

    /// ZBUFFER
    mTri3D = new cTriangulation3D<tREAL8>(aVPts,aVFaces);

    cMeshTri3DIterator  aTriIt(mTri3D);

    cSIMap_Ground2ImageAndProf aMapCamDepth(mCamPC);

    cSetVisibility aSetVis(mCamPC,mMII);

    double Infty =1e20;
    cPt2di aSzPix = mCamPC->SzPix();
    cBox3dr  aBox(cPt3dr(mMII,mMII,-Infty),cPt3dr(aSzPix.x()-mMII,aSzPix.y()-mMII,Infty));
    cDataBoundedSet<tREAL8,3>  aSetCam(aBox);
    cZBuffer aZBuf(aTriIt,aSetVis,aMapCamDepth,aSetCam,1,true);

    StdOut()<<"ZBB"<<std::endl;

    aZBuf.MakeZBuf(eZBufModeIter::ProjInit);

    aZBuf.MakeZBuf(eZBufModeIter::SurfDevlpt);
    StdOut()<<"ZBBMAKE"<<std::endl;
    ProcessNoPix(aZBuf);

    /*aZBuf.ZBufIm().DIm().ToFile("buffer-"+
                                ToStr(mIndBoxRecal.x())+"-"+
                                ToStr(mIndBoxRecal.y())+"-"+
                                LastPrefix(mNameResult)+".tif");*/

    MakeBasculeTris(aZBuf);
}


void cAppliNuageBascule::GenTFW(const cAffin2D<tREAL8> & anAff, const std::string & aNameTFW)
{
    std::ofstream aFtfw(aNameTFW.c_str());
    aFtfw.precision(10);

    aFtfw << anAff.VX().x() << "\n" << anAff.VX().y() << "\n";
    aFtfw << anAff.VY().x() << "\n" << anAff.VY().y() << "\n";
    aFtfw << anAff.Tr().x() << "\n" << anAff.Tr().y() << "\n";

    aFtfw.close();
}

cAffin2D<tREAL8> cAppliNuageBascule::ReadTFW(const std::string & aNameTFW)
{
    std::ifstream aFtfw(aNameTFW.c_str());
    std::string aline;
    std::vector< std::string > acontent;
    while(std::getline(aFtfw,aline))
    {
        acontent.push_back(aline);
    }
    aFtfw.close();

    tREAL8 gsd_x=stof(acontent.at(0));
    tREAL8 gsd_y=stof(acontent.at(3));
    tREAL8 x_ul=stof(acontent.at(4));
    tREAL8 y_ul=stof(acontent.at(5));

    acontent.clear();

    return cAffin2D<tREAL8>(cPt2dr(x_ul,y_ul),cPt2dr(gsd_x,0),cPt2dr(0,gsd_y));
}

void cAppliNuageBascule::ProcessNoPix(cZBuffer &  aZB)
{
    // comppute dual graph to have neigbouring relation between faces
    mTri3D->MakeTopo();
    const cGraphDual &  aGrD = mTri3D->DualGr() ;

    bool  GoOn = true;
    while (GoOn)  // continue as long as we get some modification
    {
        GoOn = false;
        // parse all face
        for (size_t aKF1 = 0 ; aKF1<mTri3D->NbFace() ; aKF1++)
        {
            // check for each face labled  "NoPix" if it  has a neigboor visible (or likely)
            if (aZB.ResSurfD(aKF1).mResult == eZBufRes::NoPix)
            {
                std::vector<int> aVF2;
                aGrD.GetFacesNeighOfFace(aVF2,aKF1);
                for (const auto &  aKF2 : aVF2)
                {
                    if ( ZBufLabIsOk(aZB.ResSurfD(aKF2).mResult) )
                    {
                        // Got 1 => this face is likely , and we must prolongate the global process
                        aZB.ResSurfD(aKF1).mResult =  eZBufRes::LikelyVisible;
                        GoOn = true;
                    }
                }
            }
        }
    }
}


void cAppliNuageBascule::MakeBasculeTris(cZBuffer & aZB)
{

    cPt3di aVInd;
    cPt3di aTriCorrel;
    cPt2dr anOffX(mGSD, 0.0);
    cPt2dr anOffY(0.0,-mGSD);
    cAffin2D<tREAL8> anAffinetoTarget(cPt2dr(mBoxLocTarget.P0().x(),mBoxLocTarget.P1().y()),
                                      anOffX,
                                      anOffY);
    cAffin2D<tREAL8> anInvAfftoPixel= anAffinetoTarget.MapInverse();

    cBox2dr aBoxTarget= mBoxLocTarget.CurBox();
    cPt2di aSzTarget = Pt_round_up(aBoxTarget.Sz()/mGSD);

    mImRed.DIm().Resize(aSzTarget,eModeInitImage::eMIA_Null);
    mImRed.DIm().InitCste(mInfty);

    mImMasqOut.DIm().Resize(aSzTarget,eModeInitImage::eMIA_Null);

    if (mBascCorrel)
        mImCorrelOut.DIm().Resize(aSzTarget,eModeInitImage::eMIA_Null);


    // iterate over all over triangles and ( add : confidence and occlusion info)

    for (size_t  aKF=0; aKF<mTri3D->NbFace(); aKF++)
    {
        if(aZB.ResSurfD(aKF).mResult == eZBufRes::NoPix)
        {
            //StdOut()<<"hidden"<<std::endl;
        }
        if ((aZB.ResSurfD(aKF).mResult == eZBufRes::Visible) ||
            (aZB.ResSurfD(aKF).mResult == eZBufRes::LikelyVisible) )
        {
            // apply affine transform
            tTri3dr aTri3DW = mTri3D->KthTri(aKF);

            if (mBascCorrel)
            {
                aVInd = mTri3D->KthFace(aKF);
                aTriCorrel=cPt3di(mVCorrel[aVInd[0]],mVCorrel[aVInd[1]],mVCorrel[aVInd[2]]);
            }

            cTriangle2DCompiled<tREAL8>  aTriComp(anInvAfftoPixel.Value(Proj(aTri3DW.Pt(0))),
                                                 anInvAfftoPixel.Value(Proj(aTri3DW.Pt(1))),
                                                 anInvAfftoPixel.Value(Proj(aTri3DW.Pt(2))));

            cPt3dr aElev(aTri3DW.Pt(0).z(),aTri3DW.Pt(1).z(),aTri3DW.Pt(2).z());

            std::vector<cPt2di> aVPix;
            std::vector<cPt3dr> aVW;

            aTriComp.PixelsInside(aVPix,1e-8,&aVW);

            for (size_t aK=0; aK<aVPix.size();aK++)
            {
                const cPt2di aPix = aVPix[aK];

                tREAL8 aNewZ = mMultZ * Scal(aElev,aVW[aK]);
                mImRed.DIm().SetV(aPix,aNewZ);
                mImMasqOut.DIm().SetV(aPix,1);

                if( mBascCorrel)
                {
                    mImCorrelOut.DIm().SetV(aPix, round_ni(Scal(ToR(aTriCorrel),aVW[aK])));
                }
            }
        }
    }
    // write individual images

    cDataFileIm2D aDF= cDataFileIm2D::Create(mPhProj.DPMeshDev().FullDirOut()+
                                                  "BLOC-"+
                                                  ToStr(mIndBoxRecal.x())+"-"+
                                                  ToStr(mIndBoxRecal.y())+"-"+
                                                  NameWithoutDir(mNameResult),
                                                  eTyNums::eTN_REAL8,
                                                  mImRed.DIm().Sz(),
                                                  1);

    mImRed.DIm().Write(aDF,aDF.P0());

    cDataFileIm2D aDFM= cDataFileIm2D::Create(mPhProj.DPMeshDev().FullDirOut()+
                                                   "MASQ-"+
                                                   ToStr(mIndBoxRecal.x())+"-"+
                                                   ToStr(mIndBoxRecal.y())+"-"+
                                                   NameWithoutDir(mNameResult),
                                                   eTyNums::eTN_INT1,
                                                   mImMasqOut.DIm().Sz(),
                                                   1);
    mImMasqOut.DIm().Write(aDFM,aDFM.P0());

    // write tFW DATA
    GenTFW(anAffinetoTarget, mPhProj.DPMeshDev().FullDirOut()+
                                 "BLOC-"+
                                 ToStr(mIndBoxRecal.x())+"-"+
                                 ToStr(mIndBoxRecal.y())+"-"+
                                 ChgPostix(NameWithoutDir(mNameResult),"tfw"));

    GenTFW(anAffinetoTarget, mPhProj.DPMeshDev().FullDirOut()+
                                 "MASQ-"+
                                 ToStr(mIndBoxRecal.x())+"-"+
                                 ToStr(mIndBoxRecal.y())+"-"+
                                 ChgPostix(NameWithoutDir(mNameResult),"tfw"));


    /// CORREL
    if( mBascCorrel)
    {
        cDataFileIm2D aDFC= cDataFileIm2D::Create(mPhProj.DPMeshDev().FullDirOut()+
                                                       "CORR-"+
                                                       ToStr(mIndBoxRecal.x())+"-"+
                                                       ToStr(mIndBoxRecal.y())+"-"+
                                                       NameWithoutDir(mNameResult),
                                                       eTyNums::eTN_U_INT1,
                                                       mImCorrelOut.DIm().Sz(),
                                                       1);
        mImCorrelOut.DIm().Write(aDFC,aDFC.P0());

        GenTFW(anAffinetoTarget, mPhProj.DPMeshDev().FullDirOut()+
                                     "CORR-"+
                                     ToStr(mIndBoxRecal.x())+"-"+
                                     ToStr(mIndBoxRecal.y())+"-"+
                                     ChgPostix(NameWithoutDir(mNameResult),"tfw"));
    }

}



void cAppliNuageBascule::MergeResults()
{
    mDFI2d = cDataFileIm2D::Create(mNameIm,mIsGray);
    cParseBoxInOut<2> aPBIO =  cParseBoxInOut<2>::CreateFromSize(mDFI2d,mSzTiles);

    //cPt2dr aP0 (1e10,-1e10);
    std::vector<cPt2di> aSzLocTiles;
    std::vector<cAffin2D<tREAL8>> aLocAffOut;

    /// COMPUTE GLOBAL CONTEXT

    for( const auto & PixI: aPBIO.BoxIndex())
    {
        //StdOut()<<PixI<<std::endl;        // READ CORRESPONDING LOC AFFINE TRANSFORMATIONS
        cAffin2D<tREAL8> mTrfLocBox = ReadTFW(mPhProj.DPMeshDev().FullDirOut()+
                                              "MASQ-"+
                                              ToStr(PixI.x())+"-"+
                                              ToStr(PixI.y())+"-"+
                                              ChgPostix(NameWithoutDir(mNameResult),"tfw"));

        aLocAffOut.push_back(mTrfLocBox);
        // read masq files to get the extent of the number of pixels
        // parse all images and fill global content image
        std::string aNameMasq = mPhProj.DPMeshDev().FullDirOut()+
                                "MASQ-"+
                                ToStr(PixI.x())+"-"+
                                ToStr(PixI.y())+"-"+
                                NameWithoutDir(mNameResult) ;

        cDataFileIm2D mDF = cDataFileIm2D::Create(aNameMasq,eForceGray::No);

        aSzLocTiles.push_back(mDF.Sz());

        cPt2dr aPUL = mTrfLocBox.Value(cPt2dr(0,0));
        cPt2dr aPLR = mTrfLocBox.Value(ToR(mDF.Sz()));

        mBoxGlobTarget.Add(cPt2dr(aPUL.x(),aPLR.y()));
        mBoxGlobTarget.Add(cPt2dr(aPLR.x(),aPUL.y()));
    }


    cAffin2D<tREAL8> aGlobAff(cPt2dr(mBoxGlobTarget.CurBox().P0().x(),
                                     mBoxGlobTarget.CurBox().P1().y()),
                              cPt2dr(mGSD,0),
                              cPt2dr(0,-mGSD));



    cBox2di aBoxGlobOutPix(Pt_round_up(mBoxGlobTarget.CurBox().Sz()/mGSD));

    std::string aNameBascOut = DirOfFile(mNameResult)+"Prof_"+ NameWithoutDir(mNameResult);

    std::string aNameMasqOut = DirOfFile(mNameResult)+"Masq_"+ NameWithoutDir(mNameResult);

    std::string aNameTFWGlb =  DirOfFile(mNameResult)+"Masq_"+ ChgPostix(NameWithoutDir(mNameResult),"tfw");


    GenTFW(aGlobAff,aNameTFWGlb);

    cDataFileIm2D  aDF = cDataFileIm2D::Create(aNameBascOut,
                                              eTyNums::eTN_REAL8,
                                              aBoxGlobOutPix.Sz(),
                                              1);

    cIm2D<tREAL8> aGlobIm(aDF.Sz(),aDF);
    aGlobIm.DIm().InitCste(-1e9);


    cDataFileIm2D  aDFM = cDataFileIm2D::Create(aNameMasqOut,
                                              eTyNums::eTN_U_INT1,
                                              aBoxGlobOutPix.Sz(),
                                              1);
    cIm2D<tU_INT1> aGlobMasqIm(aDFM.Sz(),aDFM);

    std::string aNameCorrelOut="";
    cIm2D<tU_INT1> aGlobCorrelIm(cPt2di(1,1));
    cDataFileIm2D aDFC=cDataFileIm2D::Empty();

    if (mBascCorrel)
    {
        aNameCorrelOut= DirOfFile(mNameResult)+"Correl_"+ NameWithoutDir(mNameResult);
        aDFC = cDataFileIm2D::Create(aNameCorrelOut,
                                                   eTyNums::eTN_U_INT1,
                                                   aBoxGlobOutPix.Sz(),
                                                   1);
        aGlobCorrelIm=cIm2D<tU_INT1>(aDFC.Sz(),aDFC);
    }

    int aK=0;
    cPt2di aP0G,aP1G;
    for( const auto & PixI: aPBIO.BoxIndex())
    {
        StdOut()<<PixI<<std::endl;
        std::string aNameProfDalle = mPhProj.DPMeshDev().FullDirOut()+
                                "BLOC-"+
                                ToStr(PixI.x())+"-"+
                                ToStr(PixI.y())+"-"+
                                NameWithoutDir(mNameResult) ;

        std::string aNameMasqDalle = mPhProj.DPMeshDev().FullDirOut()+
                                "MASQ-"+
                                ToStr(PixI.x())+"-"+
                                ToStr(PixI.y())+"-"+
                                NameWithoutDir(mNameResult) ;

        aP0G = ToI(aGlobAff.Inverse(aLocAffOut[aK].Value(cPt2dr(0,0))));
        aP1G = ToI(aGlobAff.Inverse(aLocAffOut[aK].Value(ToR(aSzLocTiles[aK]))));

        cBox2di aBoxLocInGlob(aP0G,aP1G);
        StdOut()<<"LOC "<<aBoxLocInGlob<<" "<<"GLOB "<<aBoxGlobOutPix<<std::endl;

        cIm2D<tREAL8>  aImProfDalle(aSzLocTiles[aK]);
        cIm2D<tU_INT1> aImMasqDalle(aSzLocTiles[aK]);

        aImProfDalle.Read(cDataFileIm2D::Create(aNameProfDalle,eForceGray::No),cPt2di(0,0)) ;
        aImMasqDalle.Read(cDataFileIm2D::Create(aNameMasqDalle,eForceGray::No),cPt2di(0,0)) ;

        //aImMasqDalle.DIm().Dilate(-mSzOverlap);

        cIm2D<tU_INT1> aImCorrelDalle(cPt2di(1,1));

        if (mBascCorrel)
        {
            std::string aNameCorrelDalle = mPhProj.DPMeshDev().FullDirOut()+
                                         "CORR-"+
                                         ToStr(PixI.x())+"-"+
                                         ToStr(PixI.y())+"-"+
                                         NameWithoutDir(mNameResult) ;
            aImCorrelDalle=cIm2D<tU_INT1>(aSzLocTiles[aK]);
            aImCorrelDalle.Read(cDataFileIm2D::Create(aNameCorrelDalle,eForceGray::No),
                                cPt2di(0,0)) ;
        }


        // only select masked depths  ??????
        cDataIm2D<tU_INT1> & aDIMm = aImMasqDalle.DIm();
        for (const auto & aPix: aDIMm)
        {
            if (aDIMm.GetV(aPix))
            {
                tREAL8  aSavedZ = aGlobIm.DIm().GetV(aBoxLocInGlob.P0()+aPix);
                tREAL8 aCurUpdateZ= aImProfDalle.DIm().GetV(aPix);

                if (aCurUpdateZ>aSavedZ)
                {
                    aGlobIm.DIm().SetV(aBoxLocInGlob.P0()+aPix,aCurUpdateZ);
                    aGlobMasqIm.DIm().SetV(aBoxLocInGlob.P0()+aPix,1);

                    if( mBascCorrel)
                    {
                        aGlobCorrelIm.DIm().SetV(aBoxLocInGlob.P0()+aPix,
                                                 aImCorrelDalle.DIm().GetV(aPix));
                    }
                }

            }
        }

        aK++;
    }

    // Write images

    aGlobIm.Write(aDF,cPt2di(0,0));
    aGlobMasqIm.Write(aDFM,cPt2di(0,0));

    if( mBascCorrel)
    {
        aGlobCorrelIm.Write(aDFC,cPt2di(0,0));
    }
}


/*void cAppliNuageBascule::AnalyzeSurfParams()
{
    ///< Estimates Ground Sampling Distance and Box of the image in target world coordinate system
    if (mSecCamPC)
    {
        cPt2dr aCenter= ToR(mCamPC->SzPix()) / 2.0 ;
        tREAL4 aProf = mCamPC->ImageAndDepth2Ground(cPt3dr(aCenter.x(),aCenter.y()));

    }
    else
    {

    }
}*/

int cAppliNuageBascule::ExeOnParsedBox()
{
    StdOut()<<"CURBX "<<CurBoxIn()<<std::endl;
    mImPx1= APBI_ReadIm<tREAL4>(mNameCloud2D_DepthIn);
    mImMasq1 = ReadMasqWithDef(CurBoxIn(), mNameMasq1);

    if (IsInit(&mNameCorrel))
    {
        mBascCorrel=true;
        mImCorrel = APBI_ReadIm<tU_INT1>(mNameCorrel);
    }

    if (mSecCamPC)
        MakeBasc();

    return EXIT_SUCCESS;
}


int cAppliNuageBascule::Exe()
{
    mPhProj.FinishInit();

    if (IsInit(&GSD))
        mGSD = cStrIO<tREAL8>::FromStr(GSD);

    // read camera
    mCamPC =mPhProj.ReadCamPC(APBI_NameIm(),true);


    if (mModeGeom== eModeGeom::eGEOM_EPIP)
        {
            MMVII_INTERNAL_ASSERT_strong(IsInit(&mNameSec),"should provide secondary image ");
            mSecCamPC = mPhProj.ReadCamPC(mNameSec, true);
        }



    APBI_ExecAll();

    // Merge all results of bascule

    if (!InsideParalRecall())
    {
        MergeResults();

        // REMOVE NON NECESSARY INDIVIDUAL TILES
        RemovePatternFile(mPhProj.DPMeshDev().FullDirOut()+"BLOC.*",false);
        RemovePatternFile(mPhProj.DPMeshDev().FullDirOut()+"MASQ.*",false);
        RemovePatternFile(mPhProj.DPMeshDev().FullDirOut()+"CORR.*",false);
    }

    return EXIT_SUCCESS;
}



/*  ============================================= */
/*       ALLOCATION                               */
/*  ============================================= */

tMMVII_UnikPApli Alloc_NuageBascule(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
    return tMMVII_UnikPApli(new cAppliNuageBascule(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpecNuageBascule
    (
        "NuageBascule",
        Alloc_NuageBascule,
        "Bascule of Pax/Depth maps to ground",
        {eApF::Cloud},
        {eApDT::Image},
        {eApDT::Image},
        __FILE__
        );


};
