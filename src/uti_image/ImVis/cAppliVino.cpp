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

#include "Vino.h"


#if (ELISE_X11)



std::string NameVinoPyramImage(const std::string & aDir,const std::string & aName,int aZoom)
{
     if (aZoom==1) return  aName;

     return  aDir + "Tmp-MM-Dir/" + StdPrefix(NameWithoutDir(aName)) +  "_Zoom" + ToString(aZoom) + ".tif";
}

std::string cAppli_Vino::NamePyramImage(int aZoom)
{
     return NameVinoPyramImage(mDir,mNameTiffIm,aZoom);
}

std::string  cAppli_Vino::CalculName(const std::string & aName,INT aZoom)
{
   return NameVinoPyramImage(mDir,aName,aZoom);
}

/*
*/

extern Video_Win * TheWinAffRed ;


cAppli_Vino::cAppli_Vino(int argc,char ** argv) :
    mSzIncr            (400,400),
    mNbPixMinFile      (2e6),
    mCurStats          (0),
    mTabulDynIsInit    (false)
{
    mNameXmlIn = Basic_XML_MM_File("Def_Xml_EnvVino.xml");
    if (argc>1)
    {
       mNameXmlOut =  DirOfFile(argv[1]) + "Tmp-MM-Dir/" + "EnvVino.xml";
       if (ELISE_fp::exist_file(mNameXmlOut))
          mNameXmlIn = mNameXmlOut;
    }
    EnvXml() = StdGetFromPCP(mNameXmlIn,Xml_EnvVino);

    if (argc>1)
    {
        std::string aNameFile = NameWithoutDir(argv[1]);
        mStatIsInFile = false;
        std::list<cXml_StatVino> & aLStat = Stats();
        for (std::list<cXml_StatVino>::iterator itS=aLStat.begin(); itS!=aLStat.end() ; itS++)
        {
             if (itS->NameFile() == aNameFile)
             {
                 mStatIsInFile = true;
                 mCurStats = & (*itS);
             }
        }
        if (!mStatIsInFile)
        {
            aLStat.push_back(cXml_StatVino());
            mCurStats = & aLStat.back();
            mCurStats->Type() = eDynVinoModulo;
            mCurStats->IsInit() = false ;
            mCurStats->NameFile() = aNameFile;
            mCurStats->MulDyn() = 0.5;
        }
    }


    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mNameIm, "Image", eSAM_IsExistFile),
        LArgMain()  << EAM(SzW(),"SzW",true,"size of window")
                    << EAM(ZoomBilin(),"Bilin",true,"Bilinear mode")
                    << EAM(SpeedZoomGrab(),"SZG",true,"Speed Zoom Grab")
                    << EAM(SpeedZoomMolette(),"SZM",true,"Speed Zoom Molette")
                    << EAM(LargAsc(),"WS",true,"Width Scroller")
                    << EAM(mCurStats->IntervDyn(),"Dyn",true,"Max Min value for dynamic")
    );




// Files
    mDir = DirOfFile(mNameIm);
    ELISE_fp::MkDirSvp(mDir+"Tmp-MM-Dir/");
    // MakeFileXML(EnvXml(),mNameXmlOut);



    mNameIm = NameWithoutDir(mNameIm);
    mTiffIm = new Tiff_Im(Tiff_Im::StdConvGen(mDir+mNameIm,-1,true,false));
    mNameTiffIm = mTiffIm->name();
    mTifSz = mTiffIm->sz();
    mNbPix = double(mTifSz.x) * double(mTifSz.y);
    mNbChan = mTiffIm->nb_chan();
    mCoul = ( mNbChan == 3) ;

    mSzEl = (mNbChan  * mTiffIm->bitpp()) /8.0;

    if (!mStatIsInFile)
    {
        if (! EAMIsInit(&(mCurStats->IntervDyn()  ))) 
            mCurStats->IntervDyn() = Pt2dr(0,255);
    }
    if (EAMIsInit(&(mCurStats->IntervDyn())))
    {
          mCurStats->Type() = eDynVinoMaxMin;
    }
    SaveState();
// window

    mRatioFulXY = Pt2dr(SzW()).dcbyc(Pt2dr(mTifSz));
    mRatioFul = ElMin(mRatioFulXY.x,mRatioFulXY.y);
    // mSzW = round_up(Pt2dr(mTifSz) * mRatioFul);
    mWAscH = Video_Win::PtrWStd(Pt2di(SzW().x,LargAsc()),true);
    mW =    new  Video_Win(*mWAscH,Video_Win::eBasG,SzW());
    mWAscV = new Video_Win(*mW,Video_Win::eDroiteH,Pt2di(LargAsc(),SzW().y));


    mTitle = std::string("MicMac/Vino -> ") + mNameIm;
    mW->set_title(mTitle.c_str());
    mDisp = new Video_Display(mW->disp());



    
// Scrollers

    // mVVE = new  VideoWin_Visu_ElImScr(*mW,(true?Elise_Palette(mW->prgb()):Elise_Palette(mW->pgray())),mSzIncr);
    mVVE = new  VideoWin_Visu_ElImScr(*mW,(mCoul?Elise_Palette(mW->prgb()):Elise_Palette(mW->pgray())),mSzIncr);

    mVEch.push_back(1); 
    int aSc = 1;
    while ( ((mNbPix*mSzEl)/ElSquare(aSc))  > mNbPixMinFile)
    {
        aSc *= 2;
        mVEch.push_back(aSc); 
        std::string aName = NamePyramImage(mVEch.back());

        if (! ELISE_fp::exist_file(aName))
        {
             std::string aMes = "Reduce : " +ToString(aSc);
             mW->fixed_string(Pt2dr(100,SzW().y-50),aMes.c_str() ,mWAscH->prgb()(255,0,0),true);
             TheWinAffRed = mW;
             MakeTiffRed2
             (
                 NamePyramImage(mVEch.back()/2),
                 aName,
                 mTiffIm->type_el(),
                 16,
                 false,
                 -1
             );
             TheWinAffRed = 0;
             mW->clear();
        }
    }
    mWAscH->clear();

    Tiff_Im aTifHelp = MMIcone("Help");
    mWHelp = new Video_Win(*mW,Video_Win::eSamePos,aTifHelp.sz());
    ELISE_COPY(mWHelp->all_pts(),aTifHelp.in(),mWHelp->ogray());
    mWHelp->move_to(Pt2di(mW->sz().x-mWHelp->sz().x,LargAsc()+10));

    InitMenu();
}

void cAppli_Vino::PostInitVirtual()
{
    // mVVE->SetEtalDyn(mCurStats->IntervDyn().x,mCurStats->IntervDyn().y);
    mVVE->SetChgDyn(this);
    mScr = ElPyramScroller::StdPyramide(*mVVE,mNameTiffIm,&mVEch,false,false,this);
    mScr->SetAlwaysQuick(false);
    //mScr->SetAlwaysQuickInZoom(!ZoomBilin());
    SetInterpoleMode(ZoomBilin() ? eInterpolBiLin : eInterpolPPV,false);
    // mScr = new ImFileScroller<U_INT1> (*mVVE,*mTiffIm,1.0);
    // mScr->ReInitTifFile(*mTiffIm);
    InitTabulDyn();
    mScr->set_max();
    ShowAsc();
}


void cAppli_Vino::SaveState()
{
    MakeFileXML(EnvXml(),mNameXmlOut);
}

void cAppli_Vino::Boucle()
{
    while (1)
    {
        Clik aCl = mDisp->clik_press();

        mP0Click =  aCl._pt;
        mScale0  =  mScr->sc();
        mTr0     =  mScr->tr();
        mBut0    = aCl._b;
        mCtrl0  = aCl.controled();
        mShift0  = aCl.shifted();

        // Click sur la fenetre principale 
        if (aCl._w == *mW)
        {
 
            if (mBut0==2)
               ExeClikGeom(aCl);
            if ((mBut0==4) || (mBut0==5))
            {
                ZoomMolette();
                ShowAsc();
            }

            if (mBut0==1)
            {
                GrabShowOneVal();
            }
            if (mBut0==3)
            {
                MenuPopUp();
            }
        }
        if (aCl._w == *mWAscH)
        {
             mModeGrab= eModeGrapAscX;
             mWAscH->grab(*this);
             ShowAsc();
        }
        if (aCl._w == *mWAscV)
        {
             mModeGrab= eModeGrapAscY;
             mWAscV->grab(*this);
             ShowAsc();
        }
        if (aCl._w == *mWHelp)
        {
             Help();
        }
    }
}


void cAppli_Vino::Help()
{
    mW->fill_rect(Pt2dr(0,0),Pt2dr(mW->sz()),mW->pdisc()(P8COL::white));
    
    PutFileText(*mW,Basic_XML_MM_File("HelpVino.txt"));
    mW->clik_in();
    Refresh();
}


/********************************************/
/*                                          */
/*  Lecture                                 */
/*                                          */
/********************************************/



cXml_StatVino  cAppli_Vino::StatRect(Pt2di  &aP0,Pt2di & aP1)
{
    cXml_StatVino aRes;
    CorrectRect(aP0,aP1,mTifSz);

    FillStat(aRes,rectangle(aP0,aP1),mTiffIm->in());

    return aRes;

}

void  cAppli_Vino::GrabShowOneVal()
{
    mModeGrab = eModeGrapShowRadiom;
    mW->grab(*this);
    EffaceMessageVal();
}



void  cAppli_Vino::ShowOneVal(Pt2dr aPW)
{
    Pt2di  aP = round_ni(mScr->to_user(aPW));
    Pt2di  aPp1 = aP+Pt2di(1,1);

    cXml_StatVino aStat = StatRect(aP,aPp1);

    //std::cout << "PPPPP " << aP << " " << mSom[0] << "\n";

    std::string aMesXY = " x=" + ToString(aP.x) + " y=" + ToString(aP.y);
    std::string aMesV =  " V=";
    for (int aK=0 ; aK<mNbChan; aK++)
    {
        if (aK!=0) aMesV = aMesV + " ";
        aMesV = aMesV  + StrNbChifApresVirg(aStat.Soms()[aK],3) ;
    }
        // aMesV = aMesV  + StrNbChifSign(aStat.Soms()[aK],3) + " ";
        // aMesV = aMesV  + SimplString(ToString(aStat.Soms()[aK])) + " ";

    EffaceMessageVal();



    mVBoxMessageVal.push_back(PutMessage(aPW+Pt2dr(-20,50),aMesXY,P8COL::black));

    mVBoxMessageVal.push_back(PutMessage(Pt2dr(mVBoxMessageVal.back()._p0)+Pt2dr(0,-5),aMesV,P8COL::black));

/*
    Box2di = mVBoxMessageVal.P0(
Box2di cAppli_Vino::PutMessage(Pt2dr aP0 ,const std::string & aMes,int aCoulText,Pt2dr aSzRelief,int aCoulRelief)
    mP0StrVal = Pt2di(aPW)+Pt2di(-20,30);
    mW->fixed_string(Pt2dr(mP0StrVal),aMesV.c_str(),mW->pdisc()(P8COL::black),true);
    Pt2di aSzV =  mW->SizeFixedString(aMesV);
    Pt2di aSzXY =  mW->SizeFixedString(aMesXY);

    Pt2di aP0XY = mP0StrVal + Pt2di(0,aSzXY.y+3);
    mP0StrVal.y -= aSzV.y;
    mW->fixed_string(Pt2dr(aP0XY),aMesXY.c_str(),mW->pdisc()(P8COL::black),true);


    mP1StrVal = aP0XY + Pt2di(ElMax(aSzV.x,aSzXY.x),0);
    mInitP0StrVal = true;
*/
}

#endif



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA 
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
