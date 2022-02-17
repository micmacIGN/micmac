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
#include "general/all.h"
#include "private/all.h"

/************************************************/
/*                                              */
/*             cWPointeData                     */
/*                                              */
/************************************************/

// #include "pointe.h"
class cPt2Im;
typedef enum
{
	eInexistant,
	eNonSelec,
	eSelected,
	eCur,
	eAutreImPointes // Pour ne pas perdre les pointes sur d'uatres iamgae
} eTyEtat;

class cParamPointeInit
{
      public :
	 Pt2dr Im2Norm(Pt2dr );
	 Pt2dr Norm2Im(Pt2dr );
	 virtual Pt3dr P3ofIm(Pt2dr) ; // Def Erreur Fatale
        virtual cSetPointes1Im    SetPointe(const std::string &) = 0;
	// Renvoie, pour le polygone, une liste de cibles avec des
	// pointes non initialise
	virtual std::string NamePointeInit() = 0;
	virtual std::string NamePointeInterm() = 0;
	virtual std::string NamePointeFinal() = 0;
	virtual bool SauvFinal() = 0;
	virtual bool SauvInterm() = 0;

	virtual ~cParamPointeInit();
	virtual std::string NameImageCamera() = 0;
	virtual std::string NameImagePolygone() = 0;
	virtual std::string NamePointePolygone() = 0;
	virtual const cPolygoneEtal &  Polygone() const = 0;
        virtual eTyEtat EtatDepAff() const = 0;
	virtual bool PseudoPolyg() const = 0;
	virtual std::string NamePolygone()  const = 0;
	virtual void SauvRot(ElRotation3D) const = 0;

	virtual void    ConvPointeImagePolygone(Pt2dr&);

	virtual const cPolygoneEtal::tContCible & CiblesInit() const;

	virtual  NS_ParamChantierPhotogram::cPolygoneCalib * PC() const = 0;

      protected :
         cParamPointeInit(CamStenope *);
      private :
         CamStenope * mCam;
};



     //  =========   cParamPointeInitEtalonnage ===================




class cPt1IM
{
	public :
            cPt1IM (Pt2dr aP,eTyEtat anEtat) :
               mPt   (aP),
               mEtat (anEtat)
            {
            }

            Pt2dr  mPt;
            eTyEtat  mEtat;
};

class cPt2Im
{
	public :
		cPt2Im(const cCiblePolygoneEtal * aCible,Pt2dr aP) :
                        mPtIm  (Pt2dr(-10000,-10000),eInexistant),
		        mPtPol (aP,eNonSelec),
		        pCible (aCible),
			mName  (pCible ? ToString(pCible->Ind()) : "")
		{
		}

		void Reinit()
		{
                     *this = cPt2Im(pCible,mPtPol.mPt);
		}

           cPt1IM mPtIm;
           cPt1IM mPtPol;
	   const cCiblePolygoneEtal *  pCible;
	   std::string                 mName;
};



struct cWPointeData
{
	cWPointeData(bool modeIm,Video_Win , const std::string &);
        Video_Win  mW;
        VideoWin_Visu_ElImScr mV;
	ElPyramScroller *   pScr;


};


class cPointeInit;
class cWPointe  : private cWPointeData,
                  public  EliseStdImageInteractor
{
      public :
         cWPointe(std::list<cPt2Im> &,
                  bool modeIm,
		  Video_Win,
		  const std::string &,
		  eTyEtat    aEtatDepAff,
		  cPointeInit &
		  );
	 Video_Win W();
	 // void RepCl(Clik aCl) {cout << "In : " << mName << "\n";}

	 REAL VDyn() {return mVDyn;}
	 void EtalD(REAL aV)
	 { 
		 mVDyn = aV;
		 mV.SetEtalDyn(0,round_ni(aV));
		 pScr->LoadAndVisuIm(true);
		 ShowVect();
	 }
	 void ShowVect();
	 void Refresh();
         INT GetRadiom(Pt2di aP);
         void ShowCible(NS_ParamChantierPhotogram::cCibleCalib & aCC,const Pt2dr & aP);
      private :
         void OnEndTranslate(Clik){ShowVect();}
         void OnEndScale(Clik aCl)
         {
             // EliseStdImageInteractor::OnEndScale(aCl);
             ShowVect();
         }

         std::list<cPt2Im> & mPt2Is;
	 REAL                mVDyn;
	 std::string         mName;
	 bool                mModeIm;
	 Pt2di               mSz;
         eTyEtat             mEtatDepAff;
         Tiff_Im             mTifFile;
	 cPointeInit &       mPI;

};

void PointesInitial
     (
                cParamPointeInit & aPPI,
		          Pt2di aSzWIm,
			            Pt2di aSzWPolyg
				         );



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
