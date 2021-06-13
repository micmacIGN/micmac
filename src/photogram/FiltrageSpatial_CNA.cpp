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
#include "algo_geom/delaunay_mediatrice.h"




/*******************************************************/
/*                                                     */
/*             cFichier_Trajecto                       */
/*                                                     */
/*******************************************************/


std::map<std::string,cFichier_Trajecto *> theDicoFicTraj;

cFichier_Trajecto * GetTrajFromString(const std::string & aNameFile,bool toMemo)
{
   if (DicBoolFind(theDicoFicTraj,aNameFile))
      return theDicoFicTraj[aNameFile];

   cFichier_Trajecto  aFT = StdGetObjFromFile<cFichier_Trajecto>
                            (
                                aNameFile,
                                "include/XML_GEN/SuperposImage.xml",
                                "Fichier_Trajecto",
                                "Fichier_Trajecto"
                            );
    cFichier_Trajecto * aNft = new cFichier_Trajecto(aFT);
    theDicoFicTraj[aNameFile] = aNft;
    return aNft;
}


const cPtTrajecto & cInterfChantierNameManipulateur::GetPtTrajecto
                   (
                        const cFichier_Trajecto & aFT,
                        const std::string &       aKeyAssoc,
                        const  std::string &      aNameIm
                   )
{
   std::string aKeyDic = Assoc1To1(aKeyAssoc,aNameIm,true);
   std::map<std::string,cPtTrajecto>::const_iterator iT = aFT.PtTrajecto().find(aKeyDic);
   if (iT==aFT.PtTrajecto().end())
   {
       std::cout << "For Key=" << aKeyAssoc 
                 << ", Im=" << aNameIm
                 << ", KeyIm=" << aKeyDic << "\n";
       ELISE_ASSERT(false,"GetPtTrajecto Cannot find");
   }
   return iT->second;
}


/*******************************************************/
/*                                                     */
/*             cNamePoseForFS                          */
/*                                                     */
/*******************************************************/

class  cNamePoseForFS
{
    public :
       cNamePoseForFS(const std::string& aName,Pt3dr aC);

       std::string  mName;
       Pt3dr        mC;
};

cNamePoseForFS::cNamePoseForFS
(
    const std::string& aName,
    Pt3dr aC
) :
  mName (aName),
  mC    (aC)
{
}

/*******************************************************/
/*                                                     */
/*        cPt2Of_NamePoseForFS                         */
/*                                                     */
/*******************************************************/

class cPt2Of_NamePoseForFS
{
   public :
     Pt2dr operator ()(const cNamePoseForFS & aNPF)
     {
          return Pt2dr(aNPF.mC.x,aNPF.mC.y);
     }
};

/*******************************************************/
/*                                                     */
/*        cDelaunayActSCR                              */
/*                                                     */
/*******************************************************/

class cDelaunayActSCR
{
    public :
      void operator () (const cNamePoseForFS & aS1,
                        const cNamePoseForFS & aS2)
       {
            if (mCFOR.OK_CFOR(aS1.mName,aS2.mName))
            {
               mSCR.Add(cCpleString(aS1.mName,aS2.mName));
            }
            if (mSym)
            {
                if (mCFOR.OK_CFOR(aS2.mName,aS1.mName))
                {
                    mSCR.Add(cCpleString(aS2.mName,aS1.mName));
                }
            }
       }
      void operator () (const cNamePoseForFS & aS1,
                        const cNamePoseForFS & aS2,
                        bool)
       {
            (*this)(aS1,aS2);
       }



      cDelaunayActSCR (cStdChantierRel & aSCR,bool aSym,cComputeFiltreRelOr & aCFOR) :
            mSCR (aSCR),
            mSym (aSym),
            mCFOR (aCFOR)
      {
      }

      cStdChantierRel & mSCR;
      bool              mSym;
      cComputeFiltreRelOr & mCFOR;
};



/*******************************************************/
/*                                                     */
/*           cStdChantierRel                           */
/*                                                     */
/*******************************************************/


void cStdChantierRel::ComputeFiltrageSpatial()
{
   for
   (
      std::list<cByFiltreSpatial>::const_iterator itFS=mNRD.ByFiltreSpatial().begin();
      itFS!=mNRD.ByFiltreSpatial().end();
      itFS++
   )
   {
      cComputeFiltreRelOr aCFOR(itFS->FiltreSup(),mICNM);
      std::vector<cNamePoseForFS>   aVNC;
      const std::vector<std::string> * aSet= mICNM.Get(itFS->KeySet());
      if (itFS->ByFileTrajecto().IsInit())
      {
          std::string aNF = mICNM.Dir() + itFS->ByFileTrajecto().Val();
          cFichier_Trajecto * aFT = GetTrajFromString(aNF,true);
          for (int aK=0 ; aK<int(aSet->size()) ; aK++)
          {
              const cPtTrajecto & aPtT = mICNM.GetPtTrajecto(*aFT, itFS->KeyOri(),(*aSet)[aK]);
              aVNC.push_back(cNamePoseForFS((*aSet)[aK],aPtT.Pt()));
          }
      }
      else
      {
         std::string aKO = itFS->KeyOri();
         std::string aTagC = itFS->TagCentre().Val();
         for (int aK=0 ; aK<int(aSet->size()) ; aK++)
         {
             std::string aNameOri = mICNM.Assoc1To1(aKO,(*aSet)[aK],true);

             Pt3dr aC;
             if (StdPostfix(aNameOri)==".xml")
             {
                aC = StdGetObjFromFile<Pt3dr>
                     (
                          mICNM.Dir()+ aNameOri,
                          "include/XML_GEN/ParamChantierPhotogram.xml",
                          "Centre",
                          "Pt3dr"
                     );
             }
             else
             {
                   // ElCamera *  aCam = Cam_Gen_From_File(aNameOri,"toto",true,true,&mICNM);
                  CamStenope * aCam = CamStenope::StdCamFromFile(true,aNameOri,&mICNM);
                  aC = aCam->PseudoOpticalCenter();
                   //  std::cout << "CCCCC=  " << aC << "\n"; getchar();
             }
             aVNC.push_back(cNamePoseForFS((*aSet)[aK],aC));
         }
      }
      cPt2Of_NamePoseForFS aGetP;
      cDelaunayActSCR      anAct(*this,itFS->Sym().Val(),aCFOR);
      if (itFS->FiltreDelaunay().IsInit())
      {
         const cFiltreDelaunay & aFD = itFS->FiltreDelaunay().Val();
         double aDist = aFD.DMaxDelaunay().Val();

         Delaunay_Mediatrice(&(aVNC[0]), (int)aVNC.size(),aGetP,anAct,aDist);

      }
      if (itFS->FiltreDist().IsInit())
      {
          // double aDist = itFS->FiltreDist().Val().DistMin();
          rvoisins_sortx
          (
               &(aVNC[0]),
               &(aVNC.back()),
               itFS->FiltreDist().Val().DistMax(),
               aGetP, 
               anAct
          );

      }
   }
}

/*******************************************************/
/*                                                     */
/*              cStrRelEquiv                           */
/*                                                     */
/*******************************************************/
cStrRelEquiv::cStrRelEquiv
(
    cInterfChantierNameManipulateur & aICNM,
    const cClassEquivDescripteur &    aKCE
) :
  mICNM   (aICNM),
  mKCE    (aKCE),
  mGlobS  (0),
  mCompiled(false)
{
}

void cStrRelEquiv::Compile()
{
   if (mCompiled) 
      return;

   mGlobS =  mICNM.Get(mKCE.KeySet());

   mCompiled = true;
   for (int aK=0 ; aK<int(mGlobS->size()) ; aK++)
   {
       const std::string & aName = (*mGlobS)[aK];
       std::vector<std::string>* & aV = mClasses[ValEqui(aName)];
       if (aV==0)
          aV = new std::vector<std::string>;
       aV->push_back(aName);
   }
}

const std::vector<std::string> * cStrRelEquiv::Classe(const std::string & aName) 
{
   Compile();
   const std::vector<std::string> * aRes = mClasses[ValEqui(aName)];
   if (aRes==0)
   {
      std::cout << "For Name " << aName << " Equiv " << ValEqui(aName) << "\n";
      ELISE_ASSERT(false,"cStrRelEquiv::Classe");
   }
   return aRes;
}

bool cStrRelEquiv::SameCl(const std::string & aN1,const std::string & aN2) 
{
   Compile();
  return ValEqui(aN1) == ValEqui(aN2);
}

std::string  cStrRelEquiv::ValEqui(const std::string & aName) 
{
   Compile();
   return mICNM.Assoc1To1(mKCE.KeyAssocRep(),aName,true);
}


/*******************************************************/
/*                                                     */
/*              cStrRelEquiv                           */
/*                                                     */
/*******************************************************/

   //  ------cElemComFRSE----------

cElemComFRSE::cElemComFRSE(int aNb,const std::string & aName) :
   mNb   (aNb),
   mName (aName)
{
}

int  cElemComFRSE::Nb() const {return mNb;}
const std::string & cElemComFRSE::Name() const {return mName;}



bool operator < (const cElemComFRSE & anEl1,const  cElemComFRSE & anEl2)
{
   return anEl1.Nb() > anEl2.Nb();  // On veut ranger par ordre decroissant
}


   //=====================================

cComputeFiltreRelSsEch::cComputeFiltreRelSsEch
(
    cInterfChantierNameManipulateur &  aICNM,
    const cFiltreByRelSsEch &          aFRSE
)  :
   mICNM   (aICNM),
   mFRSE   (aFRSE)
{
    const std::vector<std::string> * aSetCpl = mICNM.Get(mFRSE.KeySet());

    for (int aKCpl=0; aKCpl<int(aSetCpl->size()) ; aKCpl++)
    {
         const std::string & aNH = (*aSetCpl)[aKCpl];
         std::pair<std::string,std::string> aCpl = mICNM.Assoc2To1
                                                   (
                                                     mFRSE.KeyAssocCple(),
                                                     aNH,
                                                     false
                                                  );
          ElPackHomologue aPackH = ElPackHomologue::FromFile(mICNM.Dir()+aNH); 
          int aNb = aPackH.size();
          if (aNb>= aFRSE.SeuilBasNbPts().Val())
          {
             mElems[aCpl.first].push_back(cElemComFRSE(aNb,aCpl.second));
          }
          // std::cout << aNb << "\n";
    }
    for 
    (
         std::map<std::string,std::vector<cElemComFRSE> >::iterator itEls=mElems.begin();
         itEls!=mElems.end();
         itEls++
    )
    {
        std::vector<cElemComFRSE> & aVEls = itEls->second;
        std::sort(aVEls.begin(),aVEls.end());
        while
        (
               (! aVEls.empty())
            && (int(aVEls.size()) > aFRSE.NbMinCple().Val())
            && (aVEls.back().Nb() < aFRSE.SeuilHautNbPts().Val())
        )
        {
             aVEls.pop_back();
        }
    }
}


bool cComputeFiltreRelSsEch::OkCple(const std::string & aN1,const std::string & aN2) const
{
   
   std::map<std::string,std::vector<cElemComFRSE> >::const_iterator itEls=mElems.find(aN1);

   if (itEls==mElems.end())
      return false;

   const std::vector<cElemComFRSE> & aV=itEls->second;

   for (int aKel=0 ; aKel<int(aV.size()) ; aKel++)
      if (aV[aKel].Name() == aN2)
         return true;

   return false;
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
