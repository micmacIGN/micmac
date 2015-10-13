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
#include "MICMAC.h"
#include "im_tpl/image.h"

using namespace NS_ParamMICMAC;


class cAppliBATCH_MICMAC : public ElActionParseDir
{
    public :
      cAppliBATCH_MICMAC(int argc,char ** argv) ;
      void DoAllBatch();

   private :

      void  act(const ElResParseDir &) ;
      void  ExecCom(const std::string & aCom);

      int Verif();
      void DoOneBatch(cOneBatch &);



      cAppliMICMAC & mAPM;
      int            mBidon;
      cSectionBatch & mSB;
      cOneBatch *     mCurB;
      cElRegex *      mAutomArg1;
      std::string     mPSel;
      std::string     mDirIm;
      
};

int cAppliBATCH_MICMAC::Verif()
{
   ELISE_ASSERT(mAPM.SectionBatch().IsInit(),"No Section Batch");
   return 0;
}

cAppliBATCH_MICMAC::cAppliBATCH_MICMAC(int argc,char ** argv)  :
    mAPM       (*(cAppliMICMAC::Alloc(argc,argv,eAllocAM_Batch))),
    mBidon     (Verif()),
    mSB        (mAPM.SectionBatch().Val()),
    mCurB      (0),
    mAutomArg1 (0)
{
}

void  cAppliBATCH_MICMAC::ExecCom(const std::string & aCom)
{
   if (mSB.ExeBatch().Val())
   {
      int aCodeRetour = system(aCom.c_str());
      if (mAPM.StopOnEchecFils().Val())
      {
          ELISE_ASSERT(aCodeRetour==0,"Erreur dans processus fils");
      }
   }
   else
      std::cout << aCom << "\n";
}

void  cAppliBATCH_MICMAC::act(const ElResParseDir & aRPD)  
{
   std::string aName (aRPD.name());
   std::string aNameLoc = aName.substr(mDirIm.size(),aName.size());

   if (! mAutomArg1->Match(aNameLoc))
      return;

  for 
  (
     std::list<string>::iterator itP=mCurB->PatternCommandeBatch().begin();
     itP != mCurB->PatternCommandeBatch().end();
     itP++
  )
  {
     bool aReplace =  mAutomArg1->Replace(*itP);
     ELISE_ASSERT(aReplace,"Cannot Replace in cAppliBATCH_MICMAC::act");
     ExecCom(mAutomArg1->LastReplaced());
  }
}

void cAppliBATCH_MICMAC::DoOneBatch(cOneBatch & aBatch)
{
    mCurB = & aBatch;
    mPSel = mCurB->PatternSelImBatch();
    mAutomArg1 = new cElRegex(mPSel,15);

    ELISE_ASSERT(mAutomArg1->IsOk(),"Compile/DoOneBatch");
    mDirIm = mAPM.DirImagesInit() ;
    ElParseDir(mDirIm.c_str(),*this,1);


    delete mAutomArg1;
}

void cAppliBATCH_MICMAC::DoAllBatch()
{
   for 
   (
       std::list<cOneBatch>::iterator itB= mSB.OneBatch().begin();
       itB != mSB.OneBatch().end();
       itB++
   )
      DoOneBatch(*itB);
}


int main(int argc,char ** argv)
{
   cAppliBATCH_MICMAC anAppli(argc,argv);
   anAppli.DoAllBatch();

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
