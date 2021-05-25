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
// #include "anag_all.h"

#include "general/all.h"
#include "private/all.h"
#include "XML_GEN/all.h"

#include "im_tpl/image.h"

using namespace NS_ParamChantierPhotogram;

namespace NS_CmpMNT
{


class cOneMnt;
class cOneZoneMnt;
class cAppliCmpMnt;

class cOneMnt
{
    public :
        // aPrec assure la coherence (projection , resol ....)
	//
       cOneMnt 
       (
            const cAppliCmpMnt &,
	    const cMNT2Cmp &,
	    int aNum,
	    const cCompareMNT &,
	    cOneMnt * aPrec
       );
       void Load(const cOneZoneMnt &);
       int IdRef() const;
       Pt2di SzFile() const;
       Video_Win *   W();

       void EcartZ(Im2D_REAL4 aRes,cOneMnt &) ;
       void CorrelPente(Im2D_REAL4 aRes,cOneMnt &,const cCorrelPente &) ;

       std::string ShortName();
       void CalcVMoy(double & aZMoy,double & aPenteMoy);
       Fonc_Num Grad(); // Corrige les resolution plani
    private :
       const cAppliCmpMnt &       mAppli;
       const cMNT2Cmp &           mArgM;
       int                        mNum;
       const cCompareMNT &        mArgG;
       Video_Win *                mW;
       Im2D_REAL4                 mIm;
       TIm2D<REAL4,REAL8>         mTIm;
       std::string                mNIm;
       Tiff_Im                    mTif;
       Pt2di                      mSzFile;
       std::string                mNameXml;
       cFileOriMnt2               mFOM;
       const cOneZoneMnt *        mCurZone;
};


class cOneZoneMnt
{
    public :
       cOneZoneMnt
       (
            const cAppliCmpMnt&,
	    const cZoneCmpMnt &,
	    const cCompareMNT & aCmpMnt
       );

       const Box2di & Box() const;
       Im2D_Bits<1>   Masq() const;
       const std::string & Nom() const;

    private :
       void Init(const Box2dr &);
       void Init(const cContourPolyCM &);
       void InitFlux(Flux_Pts);

       const cAppliCmpMnt &       mAppli;
       const cZoneCmpMnt &        mArgZ;
       const cCompareMNT &        mAGlob;
       Box2di                     mBox;
       Im2D_Bits<1>               mMasq;
};


class cAppliCmpMnt
{
   public :
       static cAppliCmpMnt * StdAlloc(const std::string &);
       void DoAllCmp();
       Pt2di Sz() const;
       ~cAppliCmpMnt();
   private :
       void ShowDiff
            (
                const std::string & aMes,
	        bool  isSigned,  // Visu + calcul de biais
		double aDynVisu
	    );
       void DoOneCmp();
       cAppliCmpMnt(const cCompareMNT & aCmpMnt);


       cCompareMNT  mArg;
       Pt2di                       mSz;
       std::vector<cOneZoneMnt *>  mZones;
       std::vector<cOneMnt *>      mMnts;
       // FILE *                      mFP;
       Im2D_REAL4                  mImDiff;
       cOneZoneMnt *               mCurZ;
       cOneMnt *                   mCurRef;
       cOneMnt *                   mCurTest;
       ofstream                    mCout;
};

};



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
