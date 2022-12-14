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

extern Pt3dr Intersect_Simple(const std::vector<CamStenope *> & aVCS,const std::vector<Pt2dr> & aNPts2D);
extern double cal_Residu( Pt3dr aPInter3D , vector<CamStenope*> & aVCamInter, vector<Pt2dr> & aVPtInter);

#ifndef _MultTieP_
#define _MultTieP_


//  ===  GLOBAL a REMETTRE en Haut

bool  FileModeBin(const std::string &);

//==========================================

class cCelImTPM;
class cDicoImTPM;
class cSetPMul1ConfigTPM;
class cSetTiePMul;

  //=========================

class cCelImTPM
{
    public :
         friend class cDicoImTPM;
         friend class cSetTiePMul;

         cCelImTPM(const std::string & aNameIm,int anId);

         void *  ImTPM_GetVoidData() const;
         void    ImTPM_SetVoidData(void *);
         int     & Id() {return mId;}
         std::string  & Name() {return mNameIm;}
    private :
         std::string mNameIm;
         int         mId;
         void *      mVoidData;  // To store any usefull data
};

class cDicoImTPM
{
    public :
        friend class cSetTiePMul;
    // private :
        cCelImTPM * AddIm(const std::string &,bool &IsNew);

        std::map<std::string,cCelImTPM *> mName2Im;
        std::vector<cCelImTPM *>          mNum2Im;
};



class cSetPMul1ConfigTPM
{
    public :
       friend class cSetTiePMul;
       
       cSetPMul1ConfigTPM(const  std::vector<int> & mVIm,int aNbPts,int aNbAttr);
       void Add(const std::vector<Pt2dr> &,const std::vector<float> & aVAttr);

       Pt2dr Pt(int aKp,int aKIm)
       {
            int Adr = AddrPtIm(aKp,aKIm);
            return Pt2dr(Int2Double(mVXY[Adr]),Int2Double(mVXY[Adr+1]));
       }

       void SetPt(int aKp,int aKIm, Pt2dr & aPt)
       {
           int Adr = AddrPtIm(aKp,aKIm);
           mVXY[Adr] = Double2Int(aPt.x);
           mVXY[Adr+1] = Double2Int(aPt.y);
       }

       float Attr(int aKP,int aKAttr) const;
       const std::vector<int> & VIdIm() const;

       int   NbIm() const ;
       int   NbPts() const ;

       void *  ConfTPM_GetVoidData() const;
       void    ConfTPM_SetVoidData(void *);

       Pt2dr GetPtByImgId(int aKp, int aQueryImgID);
        
       void IntersectBundle(std::map<int, CamStenope *>&        aCams,
                            std::map<int,std::vector<Pt2dr>* >& aPtIdTr,
                            std::map<int,std::vector<int>* >&   aPtIdPId,
                            std::vector<Pt3dr>&                 aPt3D,
                            int&                                aPos); //return vectors of tracks + ids, corresponding img idx, and 3D ; fill the structures from the aPos on
       std::vector<Pt3d<double> > IntersectBundle(std::map<int, CamStenope *> aMCams); //aMCams are Camera indexed by the Id fo Image
       std::vector<Pt3d<double> > IntersectBundle(std::map<int, CamStenope *> aMCams, std::vector<double> & aVResid); // return reproj error in aVResod

    private :
       int  AddrPtIm(int aKp,int aKIm) {return 2*(aKp*mNbIm  +aKIm) ;}
        

       inline  double Int2Double(const int    & aV) { return aV*mPrec;}
       inline  int    Double2Int(const double & aV) { return round_ni(aV/mPrec);}

       double  X(int aKp,int aKIm) {return Int2Double(mVXY[AddrPtIm(aKp,aKIm)]);}
       double  Y(int aKp,int aKIm) {return Int2Double(mVXY[AddrPtIm(aKp,aKIm)+1]);}
       // static const  double ThePrec;

       std::vector<int>    mVIdIm;
       int                 mNbIm;
       int                 mNbPts;
       //  P0   X0,i1  Y0,i1 X0,i2 Y0,i2 ... P1  X1,i1  Y1,i1 X1,i2 Y1,i2
       std::vector<int>    mVXY;
       const double         mPrec;  // 1/500.0

       int                 mNbAttr;
       std::vector<float>  mVAttr;
       void *      mVoidData;  // To store any usefull data
};


class cSetTiePMul
{
    public :
        cSetTiePMul(int aNbAttrPts,const std::vector<std::string> * aVIm =0);
        void AddPts(const std::vector<int> & aNumIms,const std::vector<Pt2dr> &,const std::vector<float> & aVAttr);
        void Save(const std::string & aName);

     // ========================
        void SetFilter(const std::vector<std::string> & aVIm );
        void SetCurIms(const std::vector<std::string> & aVIm);


        static std::string StdName(cInterfChantierNameManipulateur*,const std::string aSH,const std::string & aPost,bool Bin);
        static const std::vector<std::string> * StdSetName(cInterfChantierNameManipulateur*,const std::string aSH,bool Bin);
        static const std::vector<std::string> * StdSetName_BinTxt(cInterfChantierNameManipulateur*,const std::string aSH);



        cSetPMul1ConfigTPM * OneConfigFromVI(const std::vector<INT> &);



        static cSetTiePMul * FromFiles(const std::vector<std::string> aVFiles,const std::vector<std::string>  * aFilter);

        // void Add

        void AddFile(const std::string & aName);  // Mettre en private + tard

        void ResetNbAttr(int aNbAttr);
        cDicoImTPM & DicoIm();
        cCelImTPM * CelFromName(const std::string & aName);
        cCelImTPM * CelFromInt(const int & anId);
        std::string NameFromId(const int & anId);

        const std::vector<cSetPMul1ConfigTPM *> & VPMul();
        int NbIm() const;

    private :
        
        cCelImTPM * AddIm(const std::string &,bool &IsNew);

        cDicoImTPM                       mDicoIm;
        std::vector<cSetPMul1ConfigTPM *>  mPMul;
        // Utilisee dans la conversion a partir de PHom
        std::map<std::vector<INT>,cSetPMul1ConfigTPM *>  mMapConf;

        std::vector<int>        mNumConvCur;
        std::set<std::string>*  mSetFilter;   
        int                     mNbAttr;
};

/*******************************************************************/
/*                                                                 */
/*                Conversion NEW format to OLD format              */
/*                                                                 */
/*******************************************************************/
class cPackHomol;
class cGetionStdPackHomol;

class cAppliConvertToOldFormatHom
{
    public :
        cAppliConvertToOldFormatHom(int argc,char ** argv);
        cAppliConvertToOldFormatHom(string aDir, string aPMulFile, string aOut, bool aBin, bool aIs2Way);
    private :
        void DoAll(cSetTiePMul * aSetPM,  cGetionStdPackHomol * aGes, cInterfChantierNameManipulateur *aICNM);
        std::string         mPatImage;  // Pattern image d'entree
        std::string         mSH;
        std::string         mOut;
        bool                mBin;
        cElemAppliSetFile   mEASF;      // Gestion de fichier
        const std::vector<std::string> * mFilesIm;  // vecteur nom d'image
        bool                             mExpTxt;   // Bin ou Txt export
        bool                             mIs2Way;
};

class cPackHomol
{
public :
    cPackHomol(string aIm1, string aIm2, int aId1, int aId2);
    cPackHomol(cCelImTPM * aIm1, cCelImTPM * aIm2);
    ElPackHomologue mPackDirect;
    ElPackHomologue mPackInverse;
    string mIm1;
    string mIm2;
    int    mId1;
    int    mId2;
    std::pair<int, int> mPairId;
    string CompileKey(string aHomolOut, bool isExpTxt, cInterfChantierNameManipulateur * aICNM, bool isDirect);
};


#endif //  __MultTieP_


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
aooter-MicMac-eLiSe-25/06/2007*/
