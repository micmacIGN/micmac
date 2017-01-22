/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr


    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/

#include "InitOutil.h"
#include "DrawOnMesh.h"
#include "CorrelMesh.h"
#include "Pic.h"
#include "Triangle.h"
#include <stdio.h>
#include "../../uti_phgrm/TiepTri/TiepTri.h"

void Test_Xml()
{
    cXml_TriAngulationImMaster aTriangulation = StdGetFromSI("Tri0.xml",Xml_TriAngulationImMaster);
    std::cout << "Name master " << aTriangulation.NameMaster() << "\n";
    cXml_Triangle3DForTieP aTri;
    aTri.P1() = Pt3dr(1,1,1);
    aTri.P2() = Pt3dr(1,1,2);
    aTri.P3() = Pt3dr(1,1,3);
    aTri.NumImSec().push_back(1);

    aTriangulation.Tri().push_back(aTri);
    

    aTriangulation.NameSec().push_back("toto.tif");

    MakeFileXML(aTriangulation,"Tri1.xml");
    MakeFileXML(aTriangulation,"Tri1.dmp");

    aTriangulation = StdGetFromSI("Tri1.dmp",Xml_TriAngulationImMaster);

     std::cout << "Nb tri " <<  aTriangulation.Tri().size()  << " UnPt " << aTriangulation.Tri()[1].P2() << "\n";


    exit(EXIT_SUCCESS);
}

void Test_FAST()
{
    Tiff_Im * mPicTiff = new Tiff_Im ( Tiff_Im::StdConvGen("./Test.tif",1,false));
    Pt2di mImgSz = mPicTiff->sz();
    TIm2D<double,double> * mPic_TIm2D = new TIm2D<double,double> (mPicTiff->sz());
    ELISE_COPY(mPic_TIm2D->all_pts(), mPicTiff->in(), mPic_TIm2D->out());
    //Im2D<double,double> * mPic_Im2D = new Im2D<double, double> (mPic_TIm2D->_the_im);

    Im2D_Bits<1> aMasq0  = Im2D_Bits<1>(mImgSz.x,mImgSz.y,1);
    TIm2DBits<1> TaMasq0 = TIm2DBits<1> (aMasq0);

    FastNew *aDec = new FastNew(*mPic_TIm2D , 15 , 3 , TaMasq0);
    cout<<aDec->lstPt().size()<<" pts detected "<<endl;
}

    /******************************************************************************
    The main function.
    ******************************************************************************/
int TestGiang_main(int argc,char ** argv)
{

    //Test_Xml();
    //Test_FAST();

    cout<<"********************************************************"<<endl;
    cout<<"*    TestGiang                                         *"<<endl;
    cout<<"********************************************************"<<endl;
        cout<<"dParam : param of detector : "<<endl;
        cout<<"     [FAST_Threshold]"<<endl;
        cout<<"     NO"<<endl;

        string pathPlyFileS ;
        string aTypeD="HOMOLINIT";
        string aFullPattern, aOriInput;
        string aHomolOut = "_Filtered";
        bool assum1er=false;
        int SzPtCorr = 1;int indTri=-1;double corl_seuil_glob = 0.8;bool Test=false;
        int SzAreaCorr = 5; double corl_seuil_pt = 0.9;
        double PasCorr=0.5;
        vector<string> dParam; dParam.push_back("NO");
        bool useExistHomoStruct = false;
        double aAngleF = 90;
        bool debugByClick = false;
        ElInitArgMain
                (
                    argc,argv,
                    //mandatory arguments
                    LArgMain()  << EAMC(aFullPattern, "Pattern of images",  eSAM_IsPatFile)
                    << EAMC(aOriInput, "Input Initial Orientation",  eSAM_IsExistDirOri)
                    << EAMC(pathPlyFileS, "path to mesh(.ply) file - created by Inital Ori", eSAM_IsExistFile),
                    //optional arguments
                    LArgMain()
                    << EAM(corl_seuil_glob, "corl_glob", true, "corellation threshold for imagette global, default = 0.8")
                    << EAM(corl_seuil_pt, "corl_pt", true, "corellation threshold for pt interest, default = 0.9")
                    << EAM(SzPtCorr, "SzPtCorr", true, "1->3*3,2->5*5 size of cor wind for each pt interet  default=1 (3*3)")
                    << EAM(SzAreaCorr, "SzAreaCorr", true, "1->3*3,2->5*5 size of zone autour pt interet for search default=5 (11*11)")
                    << EAM(PasCorr, "PasCorr", true, "step correlation (default = 0.5 pxl)")
                    << EAM(indTri, "indTri", true, "process one triangle")
                    << EAM(assum1er, "assum1er", true, "always use 1er pose as img master, default=0")
                    << EAM(Test, "Test", true, "Test new method - correl by XML")
                    << EAM(aTypeD, "aTypeD", true, "FAST, DIGEO, HOMOLINIT - default = HOMOLINIT")
                    << EAM(dParam,"dParam",true,"[param1, param2, ..] (selon detector - NO if don't have)", eSAM_NoInit)
                    << EAM(aHomolOut, "HomolOut", true, "default = _Filtered")
                    << EAM(useExistHomoStruct, "useExist", true, "use exist homol struct - default = false")
                    << EAM(aAngleF, "angleV", true, "limit view angle - default = 90 (all triangle is viewable)")
                    );

        if (MMVisualMode) return EXIT_SUCCESS;
        vector<double> aParamD = parse_dParam(dParam); //need to to on arg enter
        InitOutil *aChain = new InitOutil(aFullPattern, aOriInput, aTypeD,  aParamD, aHomolOut,
                                          SzPtCorr, SzAreaCorr,
                                          corl_seuil_glob, corl_seuil_pt, false, useExistHomoStruct, PasCorr, assum1er);
        aChain->initAll(pathPlyFileS);
        cout<<endl<<" +++ Verify init: +++"<<endl;
        vector<pic*> PtrPic = aChain->getmPtrListPic();
        for (uint i=0; i<PtrPic.size(); i++)
        {
            cout<<PtrPic[i]->getNameImgInStr()<<" has ";
            vector<PackHomo> packHomoWith = PtrPic[i]->mPackHomoWithAnotherPic;
            cout<<packHomoWith.size()<<" homo packs with another pics"<<endl;
            for (uint j=0; j<packHomoWith.size(); j++)
            {
                if (j!=i)
                    cout<<" ++ "<< PtrPic[j]->getNameImgInStr()<<" "<<packHomoWith[j].aPack.size()<<" pts"<<endl;
            }
        }
        vector<triangle*> PtrTri = aChain->getmPtrListTri();
        cout<<PtrTri.size()<<" tri"<<endl;
        CorrelMesh aCorrel(aChain);
        if (!Test && indTri == -1)
        {
            if (aAngleF == 90)
            {
                cout<<"All Mesh is Viewable"<<endl;
                for (uint i=0; i<PtrTri.size(); i++)
                {
                    if (useExistHomoStruct)
                        aCorrel.correlByCplExist(i);
                    else
                        aCorrel.correlInTri(i);
                }
            }
            else
            {
                cout<<"Use condition angle view"<<endl;
                for (uint i=0; i<PtrTri.size(); i++)
                {
                    if (useExistHomoStruct)
                        aCorrel.correlByCplExistWithViewAngle(i, aAngleF);
                    else
                        aCorrel.correlInTriWithViewAngle(i, aAngleF);
                }
            }
        }
        if (indTri != -1)
        {
            cout<<"Do with tri : "<<indTri<<endl;
            CorrelMesh * aCorrel = new CorrelMesh(aChain);
            if (useExistHomoStruct == false)
                aCorrel->correlInTriWithViewAngle(indTri, aAngleF, debugByClick);
            else
                aCorrel->correlByCplExistWithViewAngle(indTri, aAngleF, debugByClick);
            delete aCorrel;
        }
        if(Test)
        {



        }
        cout<<endl<<"Total "<<aCorrel.countPts<<" cpl NEW & "<<aCorrel.countCplOrg<<" cpl ORG"<<endl;
        cout<<endl;
        return EXIT_SUCCESS;
    }

int IsExtrema(TIm2D<double,double> & anIm,Pt2di aP)
{
    double aValCentr = anIm.get(aP);
    const std::vector<Pt2di> &  aVE = SortedVoisinDisk(0.5,TT_DIST_EXTREMA,true);
    int aCmp0 =0;
    for (int aKP=0 ; aKP<int(aVE.size()) ; aKP++)
    {
        int aCmp = CmpValAndDec(aValCentr,anIm.get(aP+aVE[aKP]),aVE[aKP]);
        if (aKP==0)
        {
            aCmp0 = aCmp;
            if (aCmp0==0) return 0;
        }

        if (aCmp!=aCmp0) return 0;
    }
    return aCmp0;
}

Col_Pal  ColOfType(Video_Win * mW, eTypeTieTri aType)
{
    switch (aType)
    {
          case eTTTMax : return mW->pdisc()(P8COL::red);    //max local => red
          case eTTTMin : return mW->pdisc()(P8COL::blue);   //min local => bleu
          default :;
    }
   return mW->pdisc()(P8COL::yellow);   //No Label => Jaune
}

int TestDetecteur_main(int argc,char ** argv)
{
    Pt3di mSzW;
    string aImg;
    ElInitArgMain
            (
                argc,argv,
                //mandatory arguments
                LArgMain()
                << EAMC(aImg, "img",  eSAM_None)
                << EAMC(mSzW, "mSzW", eSAM_None),
                //optional arguments
                LArgMain()
                );



    if (MMVisualMode) return EXIT_SUCCESS;
    Tiff_Im * mPicTiff = new Tiff_Im ( Tiff_Im::StdConvGen(aImg,1,false));
    Pt2di aSzIm = mPicTiff->sz();
    TIm2D<double,double> mPic_TIm2D(mPicTiff->sz());
    ELISE_COPY(mPic_TIm2D.all_pts(), mPicTiff->in(), mPic_TIm2D.out());
    Im2D<double,double> * anIm = new Im2D<double, double> (mPic_TIm2D._the_im);

    Im2D_Bits<1> aMasq0  = Im2D_Bits<1>(aSzIm.x,aSzIm.y,1);
    TIm2DBits<1> TaMasq0 = TIm2DBits<1> (aMasq0);
    /* video Win */
    Video_Win * mW_Org = 0;
    Video_Win * mW_F = 0;
    Video_Win * mW_FAC = 0; //origin, fast, fast && autocorrel
    Video_Win * mW_Final = 0;

    if (EAMIsInit(&mSzW))
    {
        if (aSzIm.x >= aSzIm.y)
        {
            double scale =  double(aSzIm.x) / double(aSzIm.y) ;
            mSzW.x = mSzW.x;
            mSzW.y = round_ni(mSzW.x/scale);
        }
        else
        {
            double scale = double(aSzIm.y) / double(aSzIm.x);
            mSzW.x = round_ni(mSzW.y/scale);
            mSzW.y = mSzW.y;
        }
        Pt2dr aZ(double(mSzW.x)/double(aSzIm.x) , double(mSzW.y)/double(aSzIm.y) );

        if (mW_Org ==0)
        {
            mW_Org = Video_Win::PtrWStd(Pt2di(mSzW.x*mSzW.z, mSzW.y*mSzW.z), true, aZ*mSzW.z);
            mW_Org->set_sop(Elise_Set_Of_Palette::TheFullPalette());
            mW_Org->set_title((aImg+"_Extr").c_str());
            ELISE_COPY(anIm->all_pts(), anIm->in(), mW_Org->ogray());
        }
        if (mW_F == 0)
        {
            mW_F = Video_Win::PtrWStd(Pt2di(mSzW.x*mSzW.z, mSzW.y*mSzW.z), true, aZ*mSzW.z);
            mW_F->set_sop(Elise_Set_Of_Palette::TheFullPalette());
            mW_F->set_title((aImg+"_FAST").c_str());
            ELISE_COPY(anIm->all_pts(), anIm->in(), mW_F->ogray());
        }
        if (mW_FAC == 0)
        {
            mW_FAC = Video_Win::PtrWStd(Pt2di(mSzW.x*mSzW.z, mSzW.y*mSzW.z), true, aZ*mSzW.z);
            mW_FAC->set_sop(Elise_Set_Of_Palette::TheFullPalette());
            mW_FAC->set_title((aImg+"_ACORREL").c_str());
            ELISE_COPY(anIm->all_pts(), anIm->in(), mW_FAC->ogray());
        }
        if (mW_Final == 0)
        {
            mW_Final = Video_Win::PtrWStd(Pt2di(mSzW.x*mSzW.z, mSzW.y*mSzW.z), true, aZ*mSzW.z);
            mW_Final->set_sop(Elise_Set_Of_Palette::TheFullPalette());
            mW_Final->set_title((aImg+"_FINAL").c_str());
            ELISE_COPY(anIm->all_pts(), anIm->in(), mW_Final->ogray());
        }
    }
    mW_Final->clik_in();


    Pt2di aP;
    std::vector<cIntTieTriInterest> aListPI;
    cFastCriterCompute * mFastCC   = cFastCriterCompute::Circle(TT_DIST_FAST);

    cCutAutoCorrelDir< TIm2D<double,double> > mCutACD (mPic_TIm2D,Pt2di(0,0),TT_SZ_AUTO_COR /2.0 ,TT_SZ_AUTO_COR);
    for (aP.x=5 ; aP.x<aSzIm.x-5 ; aP.x++)
    {
        for (aP.y=5 ; aP.y<aSzIm.y-5 ; aP.y++)
        {
            int aCmp0 =  IsExtrema(mPic_TIm2D,aP);
            if (aCmp0)
            {
                eTypeTieTri aType = (aCmp0==1)  ? eTTTMax : eTTTMin;
                bool OKAutoCorrel = !mCutACD.AutoCorrel(aP,TT_SEUIL_CutAutoCorrel_INT,TT_SEUIL_CutAutoCorrel_REEL,TT_SEUIL_AutoCorrel);
                Pt2dr aFastQual =  FastQuality(mPic_TIm2D,aP,*mFastCC,aType==eTTTMax,Pt2dr(TT_PropFastStd,TT_PropFastConsec));
                bool OkFast = (aFastQual.x > TT_SeuilFastStd) && ( aFastQual.y> TT_SeuilFastCons);
                if (OkFast && OKAutoCorrel)
                    aListPI.push_back(cIntTieTriInterest(aP,aType,aFastQual.x + 2 * aFastQual.y));
                if (mW_Org)
                {
                    mW_Org->draw_circle_loc(Pt2dr(aP),1.5,ColOfType(mW_Org, aType));    // cercle grand => extrema
                    //mW_Org->draw_circle_loc(Pt2dr(aP),0.5,mW_Org->pdisc()(OkFast ? P8COL::yellow : P8COL::cyan)); //=> cercle petit => Fast : jaune  = valid ; cyan = non valid
                }
                if (mW_F)
                {
                    mW_F->draw_circle_loc(Pt2dr(aP),1.5,ColOfType(mW_F, aType));
                    if (!OkFast)
                        mW_F->draw_circle_loc(Pt2dr(aP),1.5,mW_F->pdisc()(P8COL::cyan));
                }
                if (mW_FAC)
                {
                    mW_FAC->draw_circle_loc(Pt2dr(aP),1.5,ColOfType(mW_FAC, aType));
                    if (!OKAutoCorrel)
                        mW_FAC->draw_circle_loc(Pt2dr(aP),1.5,mW_FAC->pdisc()(P8COL::yellow));
                }
                if (mW_Final && OKAutoCorrel && OkFast)
                {
                    mW_Final->draw_circle_loc(Pt2dr(aP),1.5,ColOfType(mW_Final, aType));
                }
            }
        }

    }
    cout<<"Nb Pts :"<<aListPI.size()<<endl;
    mW_FAC->clik_in();
    return EXIT_SUCCESS;
}

