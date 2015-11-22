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
    mNbPixMinFile      (2e6)
{
    mNameXmlIn = Basic_XML_MM_File("Def_Xml_EnvVino.xml");
    if (argc>1)
    {
       mNameXmlOut =  DirOfFile(argv[1]) + "Tmp-MM-Dir/" + "EnvVino.xml";
       if (ELISE_fp::exist_file(mNameXmlOut))
          mNameXmlIn = mNameXmlOut;
    }
    EnvXml() = StdGetFromPCP(mNameXmlIn,Xml_EnvVino);



    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(mNameIm, "Image", eSAM_IsExistFile),
        LArgMain()  << EAM(SzW(),"SzW",true,"size of window")
                    << EAM(ZoomBilin(),"Bilin",true,"Bilinear mode")
                    << EAM(SpeedZoomGrab(),"SZG",true,"Speed Zoom Grab")
                    << EAM(SpeedZoomMolette(),"SZM",true,"Speed Zoom Molette")
                    << EAM(LargAsc(),"WS",true,"Width Scroller")
    );

// Files
    mDir = DirOfFile(mNameIm);
    ELISE_fp::MkDirSvp(mDir+"Tmp-MM-Dir/");
    MakeFileXML(EnvXml(),mNameXmlOut);



    mNameIm = NameWithoutDir(mNameIm);
    mTiffIm = new Tiff_Im(Tiff_Im::StdConvGen(mDir+mNameIm,-1,true,false));
    mNameTiffIm = mTiffIm->name();
    mTifSz = mTiffIm->sz();
    mNbPix = double(mTifSz.x) * double(mTifSz.y);
    mNbChan = mTiffIm->nb_chan();
    mCoul = ( mNbChan == 3) ;

    mSzEl = (mNbChan  * mTiffIm->bitpp()) /8.0;

// window

    mRatioFulXY = Pt2dr(SzW()).dcbyc(Pt2dr(mTifSz));
    mRatioFul = ElMin(mRatioFulXY.x,mRatioFulXY.y);
    // mSzW = round_up(Pt2dr(mTifSz) * mRatioFul);
    mWAscH = Video_Win::PtrWStd(Pt2di(SzW().x,LargAsc()),true);
    mW =    new  Video_Win(*mWAscH,Video_Win::eBasG,SzW());
    mWAscV = new Video_Win(*mW,Video_Win::eDroiteH,Pt2di(LargAsc(),SzW().y));


    mTitle = std::string("Vino : ") + mNameIm;
    mW->set_title(mTitle.c_str());
    mDisp = new Video_Display(mW->disp());



    
// Scrollers

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
}

void cAppli_Vino::PostInitVirtual()
{

    // ELISE_ASSERT(false,"aVEch 2 done");
    
    mVVE->SetEtalDyn(0,255);
    mScr = ElPyramScroller::StdPyramide(*mVVE,mNameTiffIm,&mVEch,false,false,this);
    mScr->SetAlwaysQuick(false);
    mScr->SetAlwaysQuickInZoom(!ZoomBilin());
    // mScr = new ImFileScroller<U_INT1> (*mVVE,*mTiffIm,1.0);
    // mScr->ReInitTifFile(*mTiffIm);
    mScr->set_max();
    ShowAsc();


if (1)
{

    mMenuMess1 = new cPopUpMenuMessage(*mW,Pt2di(300,30));
/*

    std::cout << "SZ1 " << mW->SizeFixedString("AgTYioo") << "\n";
    std::cout << "SZ2 " << mW->SizeFixedString("Ag") << "\n";


    for (int aK=0 ; aK<10  ; aK++)
    {
       mMenuMess1->ShowMessage("uiuuuuuiuououiototo",Pt2di(aK*5,aK*30),Pt3di(128,128,128));
       mW->clik_in();
       mMenuMess1->Hide();
       mW->clik_in();
    }
*/
}

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
                ShowOneVal();
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
    }
}


/********************************************/
/*                                          */
/*  Lecture                                 */
/*                                          */
/********************************************/


void  cAppli_Vino::StatRect(Pt2di  aP0,Pt2di aP1)
{
    
    aP0 = Sup(aP0,Pt2di(0,0));
    aP1 = Inf(aP1,mTifSz);

    StatFlux(rectangle(aP0,aP1));
}

void  cAppli_Vino::StatFlux(Flux_Pts aFlux)
{
   Symb_FNum aFTif(mTiffIm->in());

   ELISE_COPY
   (
        aFlux,
        Virgule(1.0,aFTif,Square(aFTif)),
        Virgule
        (
            sigma(mNb),
            sigma(mSom,mNbChan) | VMin(mMin,mNbChan) | VMax(mMax,mNbChan),
            sigma(mSom2,mNbChan)
        )
   );
}



void  cAppli_Vino::ShowOneVal()
{
    mModeGrab = eModeGrapShowRadiom;
    mW->grab(*this);
    int aR = 7;
    mScr->VisuIm(mP0StrVal-Pt2di(aR,aR),mP1StrVal+Pt2di(aR,aR),false);
}

void  cAppli_Vino::ShowOneVal(Pt2dr aPW)
{
    Pt2di  aP = round_ni(mScr->to_user(aPW));

    StatRect(aP,aP+Pt2di(1,1));

    //std::cout << "PPPPP " << aP << " " << mSom[0] << "\n";

    std::string aMes = "x=" + ToString(aP.x) + " y=" + ToString(aP.y)+  " ; V=";
    for (int aK=0 ; aK<mNbChan; aK++)
        aMes = aMes  + SimplString(ToString(mSom[aK])) + " ";

    aMes = aMes + "      ";

    mP0StrVal = Pt2di(mP0Click)+Pt2di(-20,30);

    mW->fixed_string(Pt2dr(mP0StrVal),aMes.c_str(),mW->pdisc()(P8COL::black),true);

    Pt2di aSz =  mW->SizeFixedString(aMes);
    mP0StrVal.y -= aSz.y;
    mP1StrVal = mP0StrVal + aSz;
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
