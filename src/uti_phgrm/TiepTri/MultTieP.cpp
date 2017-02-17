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
#include "MultTieP.h"

class cSetIntDyn
{
    public :
          cSetIntDyn(const  std::vector<int> &);
          bool operator < (const cSetIntDyn &);
    private :
          std::vector<int> mVI;
};


cSetIntDyn::cSetIntDyn(const  std::vector<int> & aVI) :
    mVI (aVI)
{
     std::sort(mVI.begin(),mVI.end());
}

bool cSetIntDyn::operator < (const cSetIntDyn & aSID)
{
    size_t aNb = mVI.size();
    if (aNb <  aSID.mVI.size()) 
       return true;
    if (aNb >  aSID.mVI.size()) 
       return false;
    for (size_t aK=0 ; aK<aNb ; aK++)
    {
        if ( mVI[aK] <  aSID.mVI[aK])
           return true;
        if ( mVI[aK] >  aSID.mVI[aK])
           return false;
    }
    return false;
}


/*******************************************************************/
/*                                                                 */
/*                Conversion                                       */
/*                                                                 */
/*******************************************************************/

typedef cVarSizeMergeTieP<Pt2df,cCMT_NoVal>  tMergeRat;
typedef cStructMergeTieP<tMergeRat>  tMergeStrRat;
typedef  std::map<std::string,int>  tDicNumIm;


const std::list<tMergeRat *> &  CreatePMul
                                (
                                    cVirtInterf_NewO_NameManager * aVNM,
                                    const std::vector<std::string> * aVIm
                                )
{
    
    tDicNumIm aDicoNumIm;
    for (int aKIm=0 ; aKIm<int(aVIm->size()); aKIm++)
       aDicoNumIm[(*aVIm)[aKIm]] = aKIm;

    std::string aNameCple = aVNM->NameListeCpleConnected(true);
    cSauvegardeNamedRel aLCple = StdGetFromPCP(aNameCple,SauvegardeNamedRel);

    tMergeStrRat & aMergeStruct =  *(new tMergeStrRat(aVIm->size(),false));

    for 
    (
        std::vector<cCpleString>::iterator itV=aLCple.Cple().begin();
        itV != aLCple.Cple().end();
        itV++
    )
    {
        tDicNumIm::iterator it1 = aDicoNumIm.find(itV->N1());
        tDicNumIm::iterator it2 = aDicoNumIm.find(itV->N2());
        if ((it1!=aDicoNumIm.end()) && (it2!=aDicoNumIm.end()))
        {
            std::vector<Pt2df> aVP1,aVP2;
            aVNM->LoadHomFloats(itV->N1(),itV->N2(),&aVP1,&aVP2);
            int aNum1 = it1->second;
            int aNum2 = it2->second;

            for (int aKP = 0  ; aKP<int(aVP1.size()) ; aKP++)
            {
                aMergeStruct.AddArc(aVP1[aKP],aNum1,aVP2[aKP],aNum2,cCMT_NoVal());
            }
        }
 
    }
    aMergeStruct.DoExport();
    return  aMergeStruct.ListMerged();
}

// mm3d TestLib NO_AllOri2Im DSC01.*JPG  GenOri=false

class cAppliConvertToNewFormatHom
{
    public :
        cAppliConvertToNewFormatHom(int argc,char ** argv);
    private :
        std::string         mPatImage;
        cElemAppliSetFile   mEASF;
        const std::vector<std::string> * mFilesIm;
        bool                             mDoNewOri;
        cVirtInterf_NewO_NameManager *   mVNM;
};


cAppliConvertToNewFormatHom::cAppliConvertToNewFormatHom(int argc,char ** argv) :
      mDoNewOri (true)
{
    
   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(mPatImage, "Pattern of images",  eSAM_IsPatFile),
         LArgMain()  << EAM(mDoNewOri,"DoNewOri",true,"Tuning")
   );

   mEASF.Init(mPatImage);
   mFilesIm = mEASF.SetIm();

   if (mDoNewOri)
   {
        std::string aCom =  MM3dBinFile("TestLib NO_AllOri2Im ") + QUOTE(mPatImage) + " GenOri=false ";
        System(aCom);

        std::cout << "DONE NO_AllOri2Im \n";
        // mm3d TestLib NO_AllOri2Im IMGP70.*JPG  GenOri=false 
   }

   mVNM = cVirtInterf_NewO_NameManager::StdAlloc(mEASF.mDir,"",true);
   const std::list<tMergeRat *> &  aLMR = CreatePMul  (mVNM,mFilesIm);
   std::cout << "DONE PMUL " << aLMR.size() << " \n";


   std::map<cSetIntDyn,cSetPMul1ConfigTPM *> aMapPMul;
   for (std::list<tMergeRat *>::const_iterator itMR=aLMR.begin() ; itMR!=aLMR.end() ; itMR++)
   {
        // cSetIntDyn aSet((*itMR)->VecInd());
        // if (aMapPMul[aSet]==0) 
           // aMapPMul[aSet] = new cSetPMul1ConfigTPM((*itMR)->VecInd(),0);
         // aMapPMul[aSet] ....;
          std::cout << "VecIm=" << (*itMR)->VecInd() << "\n";
         for (int aKI=1; aKI<(*itMR)->VecInd().size() ; aKI++)
         {
              ELISE_ASSERT((*itMR)->VecInd()[aKI-1]<(*itMR)->VecInd()[aKI],"Order in tMergeRat");
         }
        
   }
}


int ConvertToNewFormatHom_Main(int argc,char ** argv)
{
    cAppliConvertToNewFormatHom(argc,argv);
    return EXIT_SUCCESS;
    // :
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
aooter-MicMac-eLiSe-25/06/2007*/
