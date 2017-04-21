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


/*
*/

#include "StdAfx.h"



#define NbBuf 1000

/************************************************************/
/*                                                          */
/*                    cPointeEtalonage                      */
/*                                                          */
/************************************************************/


cPointeEtalonage::cPointeEtalonage
(
     cCiblePolygoneEtal::tInd anId,
     Pt2dr aPos,
     const cPolygoneEtal & aPolyg
) :
  mPos    (aPos),
  mCible  (& aPolyg.Cible(anId)),
  mUseIt  (true),
  mPds    (1.0)
{
}

Pt3dr cPointeEtalonage::PosTer() const
{
   return mCible->Pos();
}

Pt2dr cPointeEtalonage::PosIm() const
{
   return mPos;
}

void cPointeEtalonage::SetPosIm(Pt2dr aPos)
{
   mPos = aPos;
}

const cCiblePolygoneEtal  & cPointeEtalonage::Cible() const
{
   return * mCible;
}
bool  cPointeEtalonage::UseIt () const
{
   return mUseIt;
}

REAL cPointeEtalonage::Pds() const
{
     return mPds;
}

/************************************************************/
/*                                                          */
/*                    cSetPointes1Im                        */
/*                                                          */
/************************************************************/

bool  cSetPointes1Im::InitFromFile(const cPolygoneEtal & aPol,ELISE_fp & aFile,bool InPK1)
{
    string buf; //char buf[NbBuf]; TEST_OVERFLOW
    bool GotEOF = false;
    INT aCpt = 0;
    while (! GotEOF)
    {
        aFile.fgets(buf, GotEOF ); //aFile.fgets(buf,NbBuf,GotEOF,false); TEST_OVERFLOW
	if (GotEOF)
	{
           if (InPK1)
              ELISE_ASSERT(aCpt==0,"cSetPointes1Im::InitFromFile");
	   return false;
	}
	if ((!GotEOF) && (buf[0] != '#'))
	{
           INT anId; 
           REAL anX,anY;
           sscanf(buf.c_str(),"%d %lf %lf",&anId,&anX,&anY); // sscanf(buf,"%d %lf %lf",&anId,&anX,&anY); TEST_OVERFLOW
	   if (anId>=0)
	   {
              mPointes.push_back(cPointeEtalonage(anId,Pt2dr(anX,anY),aPol));
	   }
	   else
              return true;
	}
	aCpt++;
    }
    return false;
}

cSetPointes1Im ::cSetPointes1Im 
(
      const cPolygoneEtal & aPol,
      const std::string & aName,
      bool  SVP
)  
{
    if (SVP)
    {
       if (! ELISE_fp::exist_file(aName.c_str()))
          return;
    }

    ELISE_fp aFile(aName.c_str(),ELISE_fp::READ);
    InitFromFile(aPol,aFile,false);

}

cSetPointes1Im::cSetPointes1Im () {}



cPointeEtalonage * cSetPointes1Im ::PointeOfIdSvp(cCiblePolygoneEtal::tInd anInd)
{
    for (tCont::iterator iT=mPointes.begin(); iT!=mPointes.end() ; iT++)
      if (iT->Cible().Ind() == anInd)
         return &(*iT);

    return 0;
}

cPointeEtalonage & cSetPointes1Im ::PointeOfId(cCiblePolygoneEtal::tInd anInd)
{
    cPointeEtalonage * pRes = PointeOfIdSvp(anInd);

    ELISE_ASSERT(pRes != 0,"Cannot Find in cSetPointes1Im ::PointeOfId");
    return * pRes;
}




cSetPointes1Im::tCont & cSetPointes1Im::Pointes()
{
   return mPointes;
}

class cSetPointes1ImRMC
{
     public :
         cSetPointes1ImRMC(const std::vector<INT> & aVec) :
            mVec (aVec)
         {
         }
	 bool operator ()(const cPointeEtalonage & aPointe)
	 {
            return std::find
		   (mVec.begin(),mVec.end(),aPointe.Cible().Ind()) != mVec.end();
	 }

     private:
		const std::vector<INT> & mVec;
};


void cSetPointes1Im::RemoveCibles(const std::vector<INT> & IndToRemove)
{
	cSetPointes1ImRMC aRMC(IndToRemove);
	tCont::iterator aNewEnd = std::remove_if(mPointes.begin(),mPointes.end(),aRMC);
	mPointes.erase(aNewEnd,mPointes.end());
}

/************************************************************/
/*                                                          */
/*                    cSetNImSetPointes                     */
/*                                                          */
/************************************************************/

cSetNImSetPointes::cSetNImSetPointes 
(
      const cPolygoneEtal & aPol,
      const std::string & aName,
      bool  SVP
)  
{
    if (SVP)
    {
       if (! ELISE_fp::exist_file(aName.c_str()))
          return;
    }

    ELISE_fp aFile(aName.c_str(),ELISE_fp::READ);

    string buf; // char buf[NbBuf]; TEST_OVERFLOW
    while (true)
    {
        bool GotEOF = false;
        aFile.fgets( buf, GotEOF ); // aFile.fgets(buf,NbBuf,GotEOF,false); TEST_OVERFLOW
	if (GotEOF) 
           return;
        INT anId=0; 
        sscanf(buf.c_str(),"%d",&anId); // sscanf(buf,"%d",&anId); TEST_OVERFLOW
	if (anId<0)
           return;
	cSetPointes1Im aPointes;
	mLPointes.push_back(aPointes);
	mLPointes.back().InitFromFile(aPol,aFile,true);
    }
}

cSetNImSetPointes::tCont & cSetNImSetPointes::Pointes()
{
   return mLPointes;
}

INT cSetNImSetPointes::NbPointes()
{
   INT aRes = 0;
   for
   (
      tCont::iterator itP = mLPointes.begin();
      itP != mLPointes.end();
      itP++
   )
      aRes += (INT) itP->Pointes().size();

   return aRes;
}



/************************************************************/
/*                                                          */
/*                    cMirePolygonEtal                      */
/*                                                          */
/************************************************************/


cMirePolygonEtal::cMirePolygonEtal
(
     const std::string & aName,
     const double *      aDiams,
     INT                 aNB
)  :
    mName    (aName),
    mDiams   (aDiams),
    mNBDiam  (aNB)
{
}

bool cMirePolygonEtal::IsNegatif() const
{
    return (this==&TheNewIGN6);
}


const cMirePolygonEtal & cMirePolygonEtal::MtdMire9()
{
	return TheMTDMire9;
}

const cMirePolygonEtal & cMirePolygonEtal::IGNMire7()
{
	return TheIGNMire7;
}
const cMirePolygonEtal & cMirePolygonEtal::IGNMire5()
{
	return TheIGNMire5;
}
const cMirePolygonEtal & cMirePolygonEtal::SofianeMire3()
{
	return TheSofiane3;
}
const cMirePolygonEtal & cMirePolygonEtal::SofianeMire2()
{
	return TheSofiane2;
}

const cMirePolygonEtal & cMirePolygonEtal::SofianeMireR5()
{
	return TheSofianeR5;
}

const cMirePolygonEtal & cMirePolygonEtal::IgnMireN6()
{
   return TheNewIGN6;
}

const cMirePolygonEtal & cMirePolygonEtal::ENSGMireN6()
{
   return TheNewENSG6;
}

const cMirePolygonEtal & cMirePolygonEtal::MT0()
{
	return TheMT0;
}

const cMirePolygonEtal & cMirePolygonEtal::MTClous1()
{
	return TheMTClous1;
}


const double cMirePolygonEtal::TheIgnN6[6] = {40.0,21.5,13.0,8.0,4.5,1.0};

const double cMirePolygonEtal::TheMTD9[6] = {90.0,80.0,70.0,60.0,50.0,40.0};

const double cMirePolygonEtal::TheIGNDiams7[7] = {60.0,50.0,28.0,16.0,8.0,4.0,1.0};
const double cMirePolygonEtal::TheIGNDiams5[5] =           {28.0,16.0,9.0,5.0,1.0};
const double cMirePolygonEtal::TheSofianeDiam3[1] = {30.0};
const double cMirePolygonEtal::TheSofianeDiam2[1] = {20.0};
const double cMirePolygonEtal::TheSofianeDiamR5[5] = {34,26,18,11,8};
const double cMirePolygonEtal::TheMT0Diams[1] = {52.0};
const double cMirePolygonEtal::TheDiamMTClous1[] = {};

const double cMirePolygonEtal::TheENSG6[6] = {25.0,20.0,17.0,12.0,10.0,5.0};

//   28 16 9 5 1

cMirePolygonEtal cMirePolygonEtal::TheNewENSG6("ENSG-New-6Cercles",TheENSG6,6);
cMirePolygonEtal cMirePolygonEtal::TheNewIGN6("Ign-New-6Cercles",TheIgnN6,6);
cMirePolygonEtal cMirePolygonEtal::TheMTDMire9("MTDMire9",TheMTD9,6);

cMirePolygonEtal cMirePolygonEtal::TheIGNMire7("IGNMire7",TheIGNDiams7,7);
cMirePolygonEtal cMirePolygonEtal::TheIGNMire5("IGNMire5",TheIGNDiams5,5);

cMirePolygonEtal cMirePolygonEtal::TheSofiane3("SofianeMire3",TheSofianeDiam3,1);
cMirePolygonEtal cMirePolygonEtal::TheSofiane2("SofianeMire2",TheSofianeDiam2,1);
cMirePolygonEtal cMirePolygonEtal::TheSofianeR5("TOTO",TheSofianeDiamR5,5);

cMirePolygonEtal cMirePolygonEtal::TheMT0("Mtd0",TheMT0Diams,1);
cMirePolygonEtal cMirePolygonEtal::TheMTClous1("Mtd0",TheDiamMTClous1,0);


const cMirePolygonEtal & cMirePolygonEtal::GetFromName(const std::string & aName)
{
     if ((aName == "N6") || (aName=="Ign-New-6Cercles"))
        return TheNewIGN6;

     if ((aName == "Mtd9") || (aName=="MTDMire9"))
        return TheMTDMire9;

     if ((aName == "M7") || (aName=="IGNMire7"))
        return TheIGNMire7;
     if ((aName == "M5") || (aName=="IGNMire5"))
        return TheIGNMire5;
     if (aName == "S3")
        return TheSofiane3;
     if (aName == "S2")
        return TheSofiane2;

     if (aName == "SR5")
        return TheSofianeR5;

     if ((aName == "MT0") || (aName == "Mtd0"))
        return TheMT0;
        
     if (aName == "ENSG-New-6Cercles")
		return TheNewENSG6;
		
     if (aName == "Clous1")
        return TheMTClous1;
    std::cout << "UNKNOWN NAME =[" << aName << "]\n";
     ELISE_ASSERT(false,"Unknown name in cMirePolygonEtal::GetFromName");
     return TheIGNMire5;
}


INT cMirePolygonEtal::NbDiam() const
{
    return mNBDiam;
}

REAL cMirePolygonEtal::KthDiam(INT aK) const
{
    ELISE_ASSERT((aK>=0)&&(aK<mNBDiam),"Bad Diam Ind in cMirePolygonEtal");
    return mDiams[aK];
}

const std::string & cMirePolygonEtal::Name() const
{
	return mName;
}

/************************************************************/
/*                                                          */
/*                    cCiblePolygoneEtal                    */
/*                                                          */
/************************************************************/

cCiblePolygoneEtal::cCiblePolygoneEtal
(
) :
  mInd  (-1),
  mPos  (0,0,0),
  mMire (0),
  mQual (tQualCible(-1)),
  mCC   (0),
  mOrder (0)
{
}

cCiblePolygoneEtal::cCiblePolygoneEtal
(
    cCiblePolygoneEtal::tInd anInd,
    Pt3dr aPos,
    const cMirePolygonEtal & aMire,
    INT                      aQual,
    NS_ParamChantierPhotogram::cCibleCalib * aCC,
    int anOrder
) :
  mInd  (anInd),
  mPos  (aPos),
  mMire (&aMire),
  mQual (tQualCible(aQual)),
  mCC   (aCC),
  mOrder (anOrder)
{
      ELISE_ASSERT
      (
         (aQual>=ePerfect) && (aQual <= eBeurk),
	 "Bas Qual in cCiblePolygoneEtal::cCiblePolygoneEtal"
      );
}


NS_ParamChantierPhotogram::cCibleCalib * cCiblePolygoneEtal::CC() const
{
   return mCC;
}

Pt3dr cCiblePolygoneEtal::Pos() const
{
     return mPos;
}

void  cCiblePolygoneEtal::SetPos(Pt3dr aPos)
{
   mPos = aPos;
}

cCiblePolygoneEtal::tInd cCiblePolygoneEtal::Ind() const
{
     return mInd;
}

const cMirePolygonEtal &  cCiblePolygoneEtal::Mire() const
{
     return *mMire;
}

cCiblePolygoneEtal::tQualCible cCiblePolygoneEtal::Qual() const
{
    return mQual;
}

int cCiblePolygoneEtal::Order() const
{
   return mOrder;
}

/************************************************************/
/*                                                          */
/*                cPolygoneEtal                             */
/*                                                          */
/************************************************************/

cPolygoneEtal::cPolygoneEtal() :
  mPC (0)
{
}

NS_ParamChantierPhotogram::cPolygoneCalib * cPolygoneEtal::PC() const
{
    return mPC;
}

void cPolygoneEtal::SetPC(NS_ParamChantierPhotogram::cPolygoneCalib * aPC)
{
   mPC = aPC;
}

cPolygoneEtal::~cPolygoneEtal()
{
}

const std::list<const cCiblePolygoneEtal *> & cPolygoneEtal::ListeCible() const
{
   return mListeCible;
}

void cPolygoneEtal::LocAddCible(const cCiblePolygoneEtal * pCible)
{
    if (pCible->Order() != -1)
	mListeCible.push_back(pCible);
}

class cCmpPCibleOrder
{
    public :
       typedef  const cCiblePolygoneEtal * tCCPtr;

       bool operator() (const tCCPtr & aC1,const tCCPtr & aC2)
       {
          return aC1->Order() < aC2->Order();
       }
};

void cPolygoneEtal::PostProcess()
{
   std::vector<const cCiblePolygoneEtal *> 
        mVC(mListeCible.begin(),mListeCible.end());

   cCmpPCibleOrder aCmp;
   std::sort(mVC.begin(),mVC.end(),aCmp);

   mListeCible =  std::list<const cCiblePolygoneEtal *>
                       (mVC.begin(),mVC.end());
   
}

/************************************************************/
/*                                                          */
/*                 cPolygoneEtalImplem                      */
/*                                                          */
/************************************************************/

#include <map>
class cPolygoneEtalImplem : public cPolygoneEtal
{
	public :
		void AddCible(const cCiblePolygoneEtal &) ;
		const cCiblePolygoneEtal & Cible(cCiblePolygoneEtal::tInd) const;

	        static cPolygoneEtalImplem * FromFile(const std::string & aName);
	        static cPolygoneEtalImplem * From_File_XML
		                             (
					         const std::string & aName,
						 const cComplParamEtalPoly *
                                             );
	public :

		bool InDic(cCiblePolygoneEtal::tInd) const;
		typedef std::map<cCiblePolygoneEtal::tInd,cCiblePolygoneEtal> tDCpe;
                tDCpe   mDic;
};

bool cPolygoneEtalImplem::InDic(cCiblePolygoneEtal::tInd anInd) const
{
	return mDic.find(anInd) != mDic.end();
}

void cPolygoneEtalImplem::AddCible(const cCiblePolygoneEtal & aCible)
{
        if (InDic(aCible.Ind()))
	{
	     std::cout << "INDEXE MULT =[" << aCible.Ind()  << "]\n";
	     ELISE_ASSERT(false,"Muliple Cibl for Index in cPolygoneEtal");
	}

	mDic[aCible.Ind()] =  aCible;
	LocAddCible(&mDic[aCible.Ind()]);
}

const cCiblePolygoneEtal & cPolygoneEtalImplem::Cible
                           (cCiblePolygoneEtal::tInd anInd)  const
{
   if (! InDic(anInd))
   {
       std::cout << "INDEXE = " << anInd << "\n";
   }

   ELISE_ASSERT(InDic(anInd),"Unknonw Cible");
   return mDic.find(anInd)->second;
}


cPolygoneEtalImplem * cPolygoneEtalImplem::From_File_XML
                      (
		           const std::string & aName,
			   const cComplParamEtalPoly * aCPEP
                      )
{
    cPolygoneCalib aPC = StdGetObjFromFile<cPolygoneCalib>
                         (
			        aName,
				StdGetFileXMLSpec("ParamChantierPhotogram.xml"),
				"PolygoneCalib",
				"PolygoneCalib"
			 );

     cPolygoneEtalImplem  * aRes  = new cPolygoneEtalImplem;

     const std::vector<int> * aC2R = 0;
     if (
              aCPEP
           && aCPEP->Cible2Rech().IsInit()
           && aCPEP->Cible2Rech().Val().UseIt().Val()
        )
	aC2R = &(aCPEP->Cible2Rech().Val().Id());

     for 
     (
           std::vector<cCibleCalib>::const_iterator itC = aPC.Cibles().begin();
	   itC != aPC.Cibles().end();
	   itC++
     )
     {
           int anInd =  aC2R ?
	                IndFind(*aC2R,itC->Id()) :
			0;
	   {
              cCiblePolygoneEtal aCible
              (
                      itC->Id(),
                      itC->Position(),
                      cMirePolygonEtal::GetFromName(itC->NomType()),
                      itC->Qualite(),
                      new cCibleCalib(*itC),
		      anInd
              );
              aRes->AddCible(aCible);
           }
     }

     aRes->SetPC(new cPolygoneCalib(aPC));
     aRes->PostProcess();

     return aRes;
}




cPolygoneEtalImplem * cPolygoneEtalImplem::FromFile(const std::string & aName)
{
    cPolygoneEtalImplem  * aRes  = new cPolygoneEtalImplem;

    ELISE_fp aFile(aName.c_str(),ELISE_fp::READ);

    string buf; // char buf[NbBuf]; TEST_OVERFLOW
    bool GotEOF = false;
    while (! GotEOF)
    {
        aFile.fgets( buf, GotEOF ); // aFile.fgets(buf,NbBuf,GotEOF,false); TEST_OVERFLOW
	if ((!GotEOF) && (buf[0] != '#'))
	{
           INT anId; 
           REAL anX,anY,aZ;
           char IdMire[100];
	   INT aQual;
           sscanf(buf.c_str(),"%d %lf %lf %lf %s %d",&anId,&anX,&anY,&aZ,IdMire,&aQual); // sscanf(buf,"%d %lf %lf %lf %s %d",&anId,&anX,&anY,&aZ,IdMire,&aQual); TEST_OVERFLOW
           //cout << IdMire << " - Mire[" << anId << "] : " 
           //<< "P(" << anX << " , " << anY << " , " << aZ  << ")\n";
	   aRes->AddCible
           (
                 cCiblePolygoneEtal
                 (
                      anId,
                      Pt3dr(anX,anY,aZ),
                      cMirePolygonEtal::GetFromName(IdMire),
		      aQual,
		      0,
		      0
                 )
           );
	   //std::cout << "Done\n";
	}
    }
	
    // const cMirePolygonEtal & M7 = cMirePolygonEtal::IGNMire7();

    // aRes->AddCible(cCiblePolygoneEtal(111,Pt3dr(),M7));
    //

    
    aRes->PostProcess();
    return aRes;
}

cPolygoneEtal * cPolygoneEtal::IGN()
{
     return cPolygoneEtalImplem::FromFile("data/IGNPoly");
}

cPolygoneEtal * cPolygoneEtal::FromName
                (
		    const std::string & aName,
		    const cComplParamEtalPoly * aParam
                )
{
     if (IsPostfixed(aName) && StdPostfix(aName)== "xml")
         return cPolygoneEtalImplem::From_File_XML(aName,aParam);
     return cPolygoneEtalImplem::FromFile(aName);
}


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
