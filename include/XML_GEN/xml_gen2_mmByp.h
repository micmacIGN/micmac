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


#ifndef _ELISE_XML_GEN_MMBY_P_
#define _ELISE_XML_GEN_MMBY_P_



using namespace NS_ParamChantierPhotogram;
using namespace NS_ParamMICMAC;

class cImaMM;
class cAppliWithSetImage;
class cAppliMMByPair;

class cImaMM
{
    public :
      cImaMM(const std::string & aName,cAppliWithSetImage &);


    public :
       std::string mNameIm;
       std::string mBande;
       int         mNumInBande;
       CamStenope * mCam;
       Pt3dr        mC3;
       Pt2dr        mC2;
       Tiff_Im  &   Tiff();
       std::list<cImaMM*> mVois;
       std::string PatternOfVois(bool IncludeThis) const;
    private :
       cAppliWithSetImage &  mAppli;
       Tiff_Im  *            mPtrTiff;

};


class cAppliWithSetImage
{
   public :
      CamStenope * CamOfName(const std::string & aName);
      const std::string & Dir() const;
      int  DeZoomOfSize(double ) const;
      void operator()(cImaMM*,cImaMM*,bool);   // Delaunay call back
   protected :
      cAppliWithSetImage(int argc,char ** argv,int aFlag);
      void Develop(bool EnGray,bool En16B);

      static const int  FlagDev8BGray   = 1;
      static const int  FlagDev16BGray  = 2;

      cImaMM * ImOfName(const std::string & aName);
      void MakeStripStruct(const std::string & aPairByStrip,bool StripFirst);
      void AddDelaunayCple();
      void AddCoupleMMImSec();




      void DoPyram();

      void VerifAWSI();
      void ComputeStripPair(int);
      void AddPair(cImaMM * anI1,cImaMM * anI2);

      bool        mSym;
      bool        mShow;
      std::string mPb;
      std::string mFullName;
      std::string mDir;
      std::string mPat;
      std::string mOri;
      std::string mKeyOri;
      cInterfChantierNameManipulateur * mICNM;
      const cInterfChantierNameManipulateur::tSet * mSetIm;

      std::vector<cImaMM *> mImages;
      std::map<std::string,cImaMM *> mDicIm;
      typedef std::pair<cImaMM *,cImaMM *> tPairIm;
      typedef std::set<tPairIm> tSetPairIm;
      tSetPairIm   mPairs;
      double       mAverNbPix;
      double       mTetaBande;

   private :
      void AddPairASym(cImaMM * anI1,cImaMM * anI2);

};


template <class eType> std::list<std::string> ListOfVal(eType aValMax,const std::string& ToSub="e") // Exclue
{
    std::list<std::string> aRes;
    for (int aK=0 ; aK<int(aValMax) ; aK++)
    {
        std::string aVal = eToString((eType) aK);
        aRes.push_back(aVal.substr(ToSub.size(),std::string::npos));
    }
    return aRes;
}

#endif   // _ELISE_XML_GEN_MMBY_P_



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
