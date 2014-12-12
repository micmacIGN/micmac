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

#include "StdAfx.h"
class cOneImTDEpip;
class cAppliTDEpip;


class cOneImTDEpip
{
	public :
	   cOneImTDEpip(const std::string &,cAppliTDEpip &);
	   
       std::string ComCreateImDownScale(double aScale) const;
       std::string NameFileDownScale(double aScale) const;
	   
	   const std::string &  mMyName;
	   cAppliTDEpip &      mAppli;
	   Tiff_Im               mTifIm;
	   std::string           mNameCam;
	   ElCamera *            mCam;
	
};

class cLoaedImTDEpip
{
	 public :
	   cLoaedImTDEpip(cOneImTDEpip &,double aScale,int aSzW);
	   
	   cOneImTDEpip & mOIE;
	   std::string mNameIm;
	   Tiff_Im mTifIm;
	   Pt2di mSz;
	   TIm2D<REAL4,REAL8> mTIm;
	   Im2D<REAL4,REAL8>  mIm;
	   TIm2D<REAL4,REAL8> mTImS1;
	   Im2D<REAL4,REAL8>  mImS1;
	   TIm2D<REAL4,REAL8> mTImS2;
	   Im2D<REAL4,REAL8>  mImS2;
	   
};





class cAppliTDEpip
{
	 public :
	    cAppliTDEpip(int argc, char **argv);
	     void GenerateDownScale(int aZoomBegin,int aZoomEnd);
	     
	     void DoMatchOneScale(int aZoomBegin,int aSzW);
	
	     std::string mNameIm1;
	     std::string mNameIm2;
	     std::string mDir;
	     cInterfChantierNameManipulateur * mICNM;
	     
	     std::string mNameMasq3D;
	     cMasqBin3D *  mMasq3D;
	     cOneImTDEpip * mIm1;
	     cOneImTDEpip * mIm2;
	     int             mZoomDeb;
	     int             mZoomEnd;
	     double         mRatioIntPx;
	     int             mIntPx;

};

/*************************************************/
/***        cAppliTDEpip                       ***/
/*************************************************/

cAppliTDEpip::cAppliTDEpip(int argc, char **argv) :
    mMasq3D   (0),
    mZoomDeb  (16),
    mZoomEnd  (2),
    mRatioIntPx (0.2)
{
    ElInitArgMain
    (
       argc,argv,
       LArgMain()  << EAMC(mNameIm1, "Firt Epip Image",eSAM_IsExistFile)
                   << EAMC(mNameIm2,"Second Epip Image",eSAM_IsExistFile),
       LArgMain()  << EAM(mNameMasq3D,"Masq3D",true,"3 D Optional masq")
    );

 
      mDir = DirOfFile(mNameIm1);
      mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
      
      if (EAMIsInit(&mNameMasq3D))
         mMasq3D = cMasqBin3D::FromSaisieMasq3d(mNameMasq3D);
         
      mIm1 = new cOneImTDEpip(mNameIm1,*this);
      mIm2 = new cOneImTDEpip(mNameIm2,*this);

     mIntPx =   mIm1->mCam->SzPixel().x * mRatioIntPx;
     
	GenerateDownScale(mZoomDeb,mZoomEnd);
	
	DoMatchOneScale(mZoomDeb,2);
}

void cAppliTDEpip::GenerateDownScale(int aZoomBegin,int aZoomEnd)
{
	std::list<std::string> aLCom;
	for (int aZoom = aZoomBegin ; aZoom >= aZoomEnd ; aZoom /=2)
	{
		std::string aCom1 = mIm1->ComCreateImDownScale(aZoom);
		std::string aCom2 = mIm2->ComCreateImDownScale(aZoom);
		if (aCom1!="") aLCom.push_back(aCom1);
		if (aCom2!="") aLCom.push_back(aCom2);
    }
    		
   cEl_GPAO::DoComInParal(aLCom);
}

void cAppliTDEpip::DoMatchOneScale(int aZoom,int aSzW) 
{
	cLoaedImTDEpip aLIm1(*mIm1,aZoom,aSzW);
	cLoaedImTDEpip aLIm2(*mIm2,aZoom,aSzW);
	
}


/*************************************************/
/***        cOneImTDEpip                       ***/
/*************************************************/


cOneImTDEpip::cOneImTDEpip(const std::string & aName,cAppliTDEpip & anAppli) :
   mMyName (aName),
   mAppli  (anAppli),
   mTifIm  (mMyName.c_str()),
   mNameCam (mAppli.mICNM->Assoc1To1("NKS-Assoc-Im2Orient@-Epi",mMyName,true)),
   mCam     (Cam_Gen_From_File(mNameCam,"OrientationConique",mAppli.mICNM))
{
}


std::string cOneImTDEpip::NameFileDownScale(double aScale) const
{
	return mAppli.mDir + "Tmp-MM-Dir/Scaled-" + ToString(aScale) + "-" +mMyName;
}


std::string cOneImTDEpip::ComCreateImDownScale(double aScale) const
{
	std::string aNameRes = NameFileDownScale(aScale);
	
    if (ELISE_fp::exist_file(aNameRes)) return "";
    
    return    MM3dBinFile("ScaleIm ") 
            +  mMyName
            +  " "  + ToString(aScale)
            +  " Out="  + aNameRes;
}

/*************************************************/
/***        cLoaedIm                      	   ***/
/*************************************************/

cLoaedImTDEpip::cLoaedImTDEpip(cOneImTDEpip & aOIE,double aScale,int aSzW) :
  mOIE (aOIE),
  mNameIm (aOIE.NameFileDownScale(aScale)),
  mTifIm  (mNameIm.c_str()),
  mSz     (mTifIm.sz()),
  mTIm    (mSz),
  mIm     (mTIm._the_im),
  mTImS1  (mSz),
  mImS1   (mTImS1._the_im),
  mTImS2  (mSz),
  mImS2   (mTImS2._the_im)
{
	ELISE_COPY(mIm.all_pts(), mTifIm.in(),mIm.out());
	
	ELISE_COPY
	(
	    mIm.all_pts(),
	    rect_som(mIm.in_proj(),aSzW) / ElSquare(1+2*aSzW),
	    mImS1.out()
	);
	
	ELISE_COPY
	(
	    mIm.all_pts(),
	    rect_som(Square(mIm.in_proj()),aSzW) / ElSquare(1+2*aSzW),
	    mImS2.out()
	);
	
/**	
	ELISE_COPY
	(
	    mIm.all_pts(),
	    Min(255,10 *sqrt(Max(0,mImS2.in() - Square( mImS1.in()) ))) ,
	    mTifIm.out()
	);
	**/
	
/**	ELISE_COPY
	(
	    mIm.all_pts(), 
	    rect_max(255 - mIm.in_proj(),10), 
	    mTifIm.out()
	 );**/
}


int TDEpip_main(int argc, char **argv)
{
	cAppliTDEpip anAppli(argc,argv);
    std::cout << "TDEpip_main \n";

    return EXIT_SUCCESS;
}





/** Footer-MicMac-eLiSe-25/06/2007

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
