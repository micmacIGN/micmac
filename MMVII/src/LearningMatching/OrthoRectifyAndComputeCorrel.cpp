#include "MMVII_PCSens.h"
#include "MMVII_Geom2D.h"
#include "MMVII_DeclareCste.h"
#include <fstream>
#include <iostream>
#include "MMVII_ZBuffer.h"
#include "MMVII_Sys.h"
#include "MMVII_Radiom.h"
#include "MMVII_2Include_Tiling.h"
#include "LearnDM.h"
#include "MMVII_AllClassDeclare.h"
#include "MMVII_Interpolators.h"
#include "MMVII_PtCorrel.h"



namespace MMVII
{


namespace cNs_OrthoRectifyAndCorrel

{

class cWorldCoordinates
{

public:
    cWorldCoordinates(std::string tfw_file)
    {
        std::ifstream input_tfw(tfw_file);
        std::string aline;
        std::vector< std::string > acontent;
        while(std::getline(input_tfw,aline))
        {
            acontent.push_back(aline);
        }
        input_tfw.close();

        gsd_x=stof(acontent.at(0));
        gsd_y=stof(acontent.at(3));
        x_ul=stof(acontent.at(4));
        y_ul=stof(acontent.at(5));

        acontent.clear();
    };
    cWorldCoordinates()
    {
        gsd_x=0.0;
        gsd_y=0.0;
        x_ul=0.0;
        y_ul=0.0;
    };

    void to_world_coordinates(const cPt2dr & aPx, cPt2dr & aWPx);
    void to_pixel_coordinates(cPt2dr & aWPx, cPt2dr & aPx);

    tREAL4 gsd_x=0;
    tREAL4 gsd_y=0;
    tREAL4 x_ul=0;
    tREAL4 y_ul=0;
};


void cWorldCoordinates::to_world_coordinates(const cPt2dr & aPx, cPt2dr & aWPx)
{
    aWPx.x()=x_ul+aPx.x()*gsd_x;
    aWPx.y()=y_ul+aPx.y()*gsd_y;
}

void cWorldCoordinates::to_pixel_coordinates(cPt2dr & aWPx, cPt2dr & aPx)
{
    aPx.x()=(aWPx.x()-x_ul)/gsd_x;
    aPx.y()=(aWPx.y()-y_ul)/gsd_y;
}


class cAppliOrthoRectifyAndCorrel;

class cAppliOrthoRectifyAndCorrel : public cMMVII_Appli,
                                    public cAppliParseBoxIm<tREAL4>

{

public:
    typedef tU_INT1 tElemImage;
    typedef tREAL4 tElemDepth;

    typedef cIm2D<tElemImage> tImImage;
    typedef cIm2D<tElemDepth> tImDepth;

    typedef cDataIm2D<tElemImage> tDImImage;
    typedef cDataIm2D<tElemDepth> tDImDepth;


    cAppliOrthoRectifyAndCorrel (const std::vector<std::string> & aVargs,
                                                             const cSpecMMVII_Appli & aSpec);

    cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
    cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
    int Exe() override;
    int ExeOnParsedBox() override;
    void Rectify();
    void MakeCorrel();
    double  SimilElemByCor(const cPt2di & aP1,const cPt2di & aP2,int aSzW) const;
    double  SimilElemByCQ(const cPt2di & aP1,const cPt2di & aP2,int aSzW) const;

    // --constructed --

    cPhotogrammetricProject mPhproj;
    cSensorCamPC * mCamPC;
    //std::vector<tImImage> mVIms;

    // interpolator
    cDiffInterpolator1D *          mInterp;
    // depth or dtm as a reference surface

    std::vector<tImImage> mVOrthos;

    tImDepth mImDtm;
    tDImDepth * mDImDtm;

    tImImage mImNative;
    tDImImage * mDImNative;

    tImImage mImOrtho;
    tDImImage * mDImOrtho;

    int mSZw;
    tREAL8 mResol;
    std::string mSpecImIn;
    std::string mPatternDtm;
    std::string mNameOutCorrel;
    std::vector<std::string> mParamOrthoCorrel;
    std::string mDtmName;
    cWorldCoordinates mWorld;
    cBox2di mBoxWOk;

};



// class declaration

cAppliOrthoRectifyAndCorrel::cAppliOrthoRectifyAndCorrel(const std::vector<std::string> & aVargs,
                                                         const cSpecMMVII_Appli & aSpec):
    cMMVII_Appli(aVargs, aSpec),
    cAppliParseBoxIm<tREAL4>  (*this,eForceGray::Yes,cPt2di(2000,2000),cPt2di(50,50),false),
    mPhproj(*this),
    mCamPC(nullptr),
    mInterp(nullptr),
    mVOrthos(std::vector<tImImage>()),
    mImDtm (cPt2di(1,1)),
    mDImDtm(nullptr),
    mImNative(cPt2di(1,1)),
    mDImNative(nullptr),
    mImOrtho(cPt2di(1,1)),
    mDImOrtho(nullptr),
    mSZw(1),
    mResol(0.5),
    //mSpecImIn(""),
    //mPatternDtm(""),
    mNameOutCorrel("Correl_"),
    mWorld(cWorldCoordinates()),
    mBoxWOk(cBox2di::Empty())
{
}


cCollecSpecArg2007 & cAppliOrthoRectifyAndCorrel::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
           << Arg2007(mSpecImIn,"Pattern/file for images",{{eTA2007::MPatFile,"0"},{eTA2007::FileDirProj}})
           << Arg2007(mPatternDtm,"Pattern of dtms ",{{eTA2007::MPatFile,"1"}})
           << mPhproj.DPOrient().ArgDirInMand()
        ;
}


cCollecSpecArg2007 & cAppliOrthoRectifyAndCorrel::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
    return anArgOpt
           << AOpt2007(mNameOutCorrel,"CorrelMap" ,"Correl Map image Name")
           << AOpt2007(mParamOrthoCorrel,"ParamCorrel","Paramaters for adj Lidar/Phgr [[SzwCorrel,Interp,Resolution en (m)]*]")
        ;
}


/*int cAppliOrthoRectifyAndCorrel::ExeOnParsedBox()
{
    //StdOut()<<"OOO "<<mDtmName.replace(mDtmName.find("tif"),3,"tfw")<<std::endl;
    // Boxed read of dtm
    mImDtm = APBI_ReadIm<tElemDepth>(mDtmName);
    mDImDtm = &(mImDtm.DIm());

    // orthorectify
    //cPt2di aPix;
    cPt2dr aWPx;
    for(const auto & aPix: *mDImDtm)
    {
            // to world coordinates
            mWorld.to_world_coordinates(ToR(aPix),aWPx);
            cPt3dr aWP3D(aWPx.x(),aWPx.y(),mDImDtm->GetV(aPix));
            if( mCamPC->IsVisible(aWP3D))
            {
                cPt2dr aPInCam= mCamPC->Ground2Image(aWP3D);
                if(mDImNative->InsideInterpolator(*mInterp,aPInCam,1.0))
                {
                    auto aVGr=mDImNative->GetValueAndGradInterpol(*mInterp,aPInCam);
                    mDImOrtho->VI_SetV(aPix,round_ni(aVGr.first));
                }
            }
    }
    APBI_WriteIm("Ortho_"+mCamPC->NameImage(),mImOrtho);
    return EXIT_SUCCESS;
}*/




/*cBox2di cAppliOrthoRectifyAndCorrel::BoxInGeomImage( cSensorCamPC & aCam, cBox2di & aBoxTerrain)
{
    cPt2di aP0 = aBoxTerrain.P0();
    cPt2di aP1 = aBoxTerrain.P1();


}*/

int cAppliOrthoRectifyAndCorrel::ExeOnParsedBox()
{
    /*StdOut()<<CurBoxIn().P0()<<CurBoxIn().P1()<<std::endl;
    mBoxWOk = CurBoxInLoc().Dilate(-mSZw); // Box of pix with window include => erosion of cur box
    StdOut()<<mBoxWOk.P0()<<mBoxWOk.P1()<<std::endl;
    MakeCorrel();
    return EXIT_SUCCESS;*/
    mImDtm = APBI_ReadIm<tREAL4>(mDtmName);
    mBoxWOk = CurBoxInLoc().Dilate(-mSZw);
    return EXIT_SUCCESS;
}




double   cAppliOrthoRectifyAndCorrel::SimilElemByCor(const cPt2di & aP1,const cPt2di & aP2,int aSzW) const
{
    if ( (!mBoxWOk.Inside(aP1)) ||  (!mBoxWOk.Inside(aP2)) )
        return 1.0;
    cMatIner2Var<double>  aMat;

    const tDImImage & aI1=     mVOrthos[0].DIm();
    const tDImImage & aI2 =    mVOrthos[1].DIm();
    for (const auto & aDP : cRect2::BoxWindow(aSzW))
    {
        aMat.Add(aI1.GetV(aP1+aDP),aI2.GetV(aP2+aDP));
    }
    return aMat.Correl(); //(1- aMat.Correl())/2.0;
}

double   cAppliOrthoRectifyAndCorrel::SimilElemByCQ(const cPt2di & aP1,const cPt2di & aP2,int aSzW) const
{
    if ( (!mBoxWOk.Inside(aP1)) ||  (!mBoxWOk.Inside(aP2)) )
        return 1.0;
    const tDImImage & aI1=     mVOrthos[0].DIm();
    const tDImImage & aI2 =    mVOrthos[1].DIm();

    double aVC1 = aI1.GetV(aP1);
    double aVC2 = aI2.GetV(aP2);

    double aSomDif = 0.0;
    double aNb     = 0.0;
    for (const auto & aDP : cRect2::BoxWindow(aSzW))
    {
        double aV1 = aI1.GetV(aP1+aDP);
        double aV2 = aI2.GetV(aP2+aDP);
        double aR1 = NormalisedRatioPos(aV1,aVC1);
        double aR2 = NormalisedRatioPos(aV2,aVC2);

        aSomDif += std::abs(aR1-aR2);
        aNb++;
    }
    double aRes = std::min(1.0,aSomDif / aNb);
    aRes = pow(aRes,0.25);

    return aRes;
}



void   cAppliOrthoRectifyAndCorrel::MakeCorrel()
{
    cIm2D<tREAL4>  aImCorr(mVOrthos[0].DIm().Sz());
    mBoxWOk = cBox2di(cPt2di(0,0),aImCorr.DIm().Sz()).Dilate(-mSZw);

    for (const auto & aPix : aImCorr.DIm())
        aImCorr.DIm().SetV(aPix,SimilElemByCor(aPix,aPix,mSZw));
        //aImCorr.DIm().SetVTrunc(aPix,255.0*(1-SimilElemByCor(aPix,aPix,mSZw)));
    //APBI_WriteIm(mNameOutCorrel+mCamPC->NameImage(),aImCorr);
    aImCorr.DIm().ToFile(mNameOutCorrel+mCamPC->NameImage());
}





void cAppliOrthoRectifyAndCorrel::Rectify()
{

    mImDtm = cIm2D<tElemDepth>::FromFile(mDtmName);
    mDImDtm = &(mImDtm.DIm());

    // orthorectify
    //cPt2di aPix;
    cPt2dr aWPx;
    for(const auto & aPix: *mDImDtm)
    {
        // to world coordinates
        mWorld.to_world_coordinates(ToR(aPix),aWPx);
        cPt3dr aWP3D(aWPx.x(),aWPx.y(),mDImDtm->GetV(aPix));
        if( mCamPC->IsVisible(aWP3D))
        {
            cPt2dr aPInCam= mCamPC->Ground2Image(aWP3D);
            if(mDImNative->InsideInterpolator(*mInterp,aPInCam,1.0))
            {
                auto aVGr=mDImNative->GetValueAndGradInterpol(*mInterp,aPInCam);
                mDImOrtho->VI_SetV(aPix,round_ni(aVGr.first));
            }
        }
    }
    //APBI_WriteIm("Ortho_"+mCamPC->NameImage(),mImOrtho);
    mDImOrtho->ToFile("Ortho_"+mCamPC->NameImage());
}

int cAppliOrthoRectifyAndCorrel::Exe()
{
    // read images and dtm
    mPhproj.FinishInit();

    std::vector<std::string> aParamInt {"Tabul","1000","SinCApod","10","10"};
    if (mParamOrthoCorrel.size()>0)
    {
        // mSzW
        mSZw=cStrIO<int>::FromStr(mParamOrthoCorrel.at(0));
    }

    if ((mParamOrthoCorrel.size()>1) && (!mParamOrthoCorrel.at(1).empty()))
    {
        // user supplied interpolator parameters
        aParamInt = Str2VStr(mParamOrthoCorrel.at(2));
    }
    mInterp  = cDiffInterpolator1D::AllocFromNames(aParamInt);


    if (mParamOrthoCorrel.size()>2)
        mResol=cStrIO<tREAL8>::FromStr(mParamOrthoCorrel.at(2));


    std::vector<std::string> aVecIms= VectMainSet(0);
    std::vector<std::string> aVecDtms= VectMainSet(1);

    for( const auto & aDtmName: aVecDtms)
    {
        // read dtm
        StdOut()<<aDtmName<<std::endl;
        mDtmName= aDtmName;
        std::string aWorldFile=ChgPostix(aDtmName,"tfw") ;
        mWorld= cWorldCoordinates(aWorldFile);
        this->mNameIm=mDtmName;
        for (const auto & aImName: aVecIms)
        {
            // initialize all camers
            mCamPC=mPhproj.ReadCamPC(aImName,true);

            StdOut()<<mCamPC->NameImage()<<std::endl;

            // read full image to orthorectify
            mImNative = tImImage::FromFile(mCamPC->NameImage());
            mDImNative = &(mImNative.DIm());

            // Intialize Rectified Image to save

            mImOrtho=tImImage(cDataFileIm2D::Create(this->mNameIm,this->mIsGray).Sz(),nullptr,eModeInitImage::eMIA_Null);
            mDImOrtho= &(mImOrtho.DIm());

            // call parallel partioning of dtm and computing resolution
            /*if (RunMultiSet(0,0))
                return ResultMultiSet();
            APBI_ExecAll();*/
            Rectify();
            mVOrthos.push_back(mImOrtho);
        }

        //APBI_ExecAll();
        MakeCorrel();
    }



    delete mInterp;
    mCamPC=nullptr;
    mDImDtm=nullptr;
    mDImNative=nullptr;
    mDImOrtho=nullptr;

    return EXIT_SUCCESS;
}


};


using namespace cNs_OrthoRectifyAndCorrel;

tMMVII_UnikPApli Alloc_OrthoRectifyAndCorrel(const std::vector<std::string> &  aVArgs,
                                             const cSpecMMVII_Appli & aSpec)
{
    return tMMVII_UnikPApli(new cAppliOrthoRectifyAndCorrel(aVArgs,aSpec));
}

cSpecMMVII_Appli TheSpecOrthoRectifyAndCorrel
    (
        "OrthpoRectify",
        Alloc_OrthoRectifyAndCorrel,
        "Ortho rectify image on provided dtm",
        {eApF::Match},
        {eApDT::Image},
        {eApDT::ToDef},
        __FILE__
        );
};
