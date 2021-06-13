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

#include "RTI.h"

const std::string cAppli_RTI::ThePrefixReech = "Tmp-MM-Dir/RTI_REECH_";


void cAppli_RTI::CreateSuperpHom()
{
    std::string aCom = MM3dBinFile_quotes("TestLib")
                                     + " AllReechHom "
                                     +  mParam.MasterIm() 
                                     +  " " + mParam.Pattern()
                                     +  " " +  ThePrefixReech ;

   System (aCom);

}

void cAppli_RTI::CreatHom()
{
    // int aSzTapioca = 2000;
    std::string aCom =    MM3dBinFile_quotes("Tapioca") +  " All " 
                       +  mParam.Pattern()  + " "
                       +  ToString(mParam.SzHom().Val()) +  " "
                       +  QUOTE("Pat2=" + mParam.Pattern()) ;

   System (aCom);
}



cAppli_RTI::cAppli_RTI(const std::string & aFullNameParam,eModeRTI aMode,const std::string & aNameI2) :
   mTest      (true),
   mNameImMed (ThePrefixReech + "Mediane.tif"),
   mNameImGx  (ThePrefixReech + "Gx.tif"),
   mNameImGy  (ThePrefixReech + "Gy.tif")
{
    mFullNameParam = aFullNameParam;

    
    SplitDirAndFile(mDir,mNameParam,mFullNameParam);
    mParam = StdGetFromSI(mFullNameParam,Xml_ParamRTI);

    mICNM= cInterfChantierNameManipulateur::BasicAlloc(mDir);
    mOriMaster = 0;

    if (aNameI2!="")
    {
       mParam.Pattern() = aNameI2;
    }

    mWithRecal = mParam.SzHom().IsInit();

    if (mWithRecal)
    {
       // CreatHom();
       CreateSuperpHom();
    }

    mEASF.Init(mDir+mParam.Pattern());
    const cInterfChantierNameManipulateur::tSet *  aSetIm = mEASF.SetIm();

    mMasterIm = new cOneIm_RTI_Master(*this,mParam.MasterIm());
    mMasterIm->DoImReduite();
    mVIms.push_back(mMasterIm);
    mDicoIm[mMasterIm->Name()] =  mMasterIm;

    for (size_t aKI = 0; aKI < aSetIm->size(); aKI++)
    {
         std::string aName = (*aSetIm)[aKI];
         if (aName != mParam.MasterIm())
         {
             cOneIm_RTI_Slave * aNewIm = new cOneIm_RTI_Slave(*this,aName);
             mVIms.push_back(aNewIm);
             mVSlavIm.push_back(aNewIm);
             aNewIm->DoImReduite();
             mDicoIm[aNewIm->Name()] =  aNewIm;
         }
    }

    for (std::list<cXml_RTI_Im>::iterator itI=mParam.RTI_Im().begin() ; itI!=mParam.RTI_Im().end() ; itI++)
    {
        const std::string & aName = itI->Name();
        cOneIm_RTI* anIm= mDicoIm[aName];
        ELISE_ASSERT(anIm!=0,"No Image from cXml_RTI_Im");
 
        anIm->SetXml(&(*itI));
    }


}

const cXml_ParamRTI & cAppli_RTI::Param() const { return mParam; }
const std::string &   cAppli_RTI::Dir()   const {return mDir;}
cOneIm_RTI_Master *   cAppli_RTI::Master() {return mMasterIm;}
bool  cAppli_RTI::WithRecal() const { return mWithRecal; }

cOneIm_RTI_Slave * cAppli_RTI::UniqSlave()
{
   ELISE_ASSERT(mVSlavIm.size()==1,"cAppli_RTI::UniqSlave");
   return mVSlavIm[0];
}




int  Gen_RTI_main(int argc,char ** argv,eModeRTI aMode)
{
    std::string aFullNameParam,aPat="";
    ElInitArgMain
    (
          argc,argv,
          LArgMain()  << EAMC(aFullNameParam,"Name of Xml Param", eSAM_IsExistFile),
          LArgMain()  << EAM(aPat,"Pat",true,"Pattern to replace existing Pattern in xml file",eSAM_IsExistFile) 
    );
    
    if (aMode==eRTI_RecalBeton_1Im)
    {
         ELISE_ASSERT(EAMIsInit(&aPat),"Gen_RTI_main Pat non init");
    }
    cAppli_RTI anAppli(aFullNameParam,aMode,aPat);

    if (aMode==eRTI_RecalBeton_1Im)
    {
        anAppli.DoOneRecalRadiomBeton();
    }

    return EXIT_SUCCESS;
}
int  RTI_main(int argc,char ** argv)
{
      return Gen_RTI_main(argc,argv,eRTI_Test);
}

int  RTI_RecalRadionmBeton_main(int argc,char ** argv)
{
      return Gen_RTI_main(argc,argv,eRTI_RecalBeton_1Im);
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
