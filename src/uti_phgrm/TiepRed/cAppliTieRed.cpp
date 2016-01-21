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

#include "TiepRed.h"


/**********************************************************************/
/*                                                                    */
/*                         cAppliTiepRed                              */
/*                                                                    */
/**********************************************************************/


cAppliTiepRed::cAppliTiepRed(int argc,char **argv)  :
     mFilesIm                 (0),
     mPrec2Point              (5.0),
     mThresholdPrecMult       (2.0),  // Multiplier of Mediane Prec, can probably be stable
     mThresholdNbPts2Im       (3),
     mThresholdTotalNbPts2Im  (10),
     mSzTile                  (1600),
     mDistPMul                (200.0),
     mCallBack                (false)
{
   // Read parameters 
   MMD_InitArgcArgv(argc,argv);
   ElInitArgMain
   (
         argc,argv,
         LArgMain()  << EAMC(mPatImage, "Pattern of images",  eSAM_IsPatFile),
         LArgMain()  << EAM(mCalib,"OriCalib",true,"Calibration folder if any")
                     << EAM(mPrec2Point,"Prec2P",true,"Threshold of precision for 2 Points")
                     << EAM(mKBox,"KBox",true,"Internal use")
                     << EAM(mSzTile,"SzTile",true,"Size of Tiles in Pixel")
                     << EAM(mDistPMul,"DistPMul",true,"Typical dist between pmult")
   );
   // if mKBox was set, we are not the master call (we are the "subcommand")
   mCallBack = EAMIsInit(&mKBox);
   mDir = DirOfFile(mPatImage);
   mICNM = cInterfChantierNameManipulateur::BasicAlloc(mDir);
   // Correct orientation (for example Ori-toto => toto)
   if (EAMIsInit(&mCalib))
   {
      StdCorrecNameOrient(mCalib,mDir);
   }
   
   if (mCallBack)  // Subcommand mode, initialise set of file for Param_K.xml
   {
       mXmlParBox = StdGetFromPCP(NameParamBox(mKBox,true),Xml_ParamBoxReducTieP);
       mBoxLoc = mXmlParBox.Box();
       mFilesIm = &(mXmlParBox.Ims());
   std::cout << "=======================   KBOX=" << mKBox << "  ===================\n";
   }
   else  // Master command, initialise from pattern
   {
       cElemAppliSetFile anEASF(mPatImage);
       // anEASF.Init(mPatImage);
       mFilesIm = anEASF.SetIm();
   }


   mSetFiles = new std::set<std::string>(mFilesIm->begin(),mFilesIm->end());
   std::cout << "## Get Nb Images " <<  mFilesIm->size() << "\n";


   mNM = cVirtInterf_NewO_NameManager::StdAlloc(mDir,mCalib);

   std::vector<double> aVResol;
   Pt2dr aPInf( 1E50, 1E50);
   Pt2dr aPSup(-1E50,-1E50);

   // Parse the images 
   for (int aKI = 0 ; aKI<int(mFilesIm->size()) ; aKI++)
   {
       const std::string & aNameIm = (*mFilesIm)[aKI];
        // Get the camera created by Martini 
       CamStenope * aCS = mNM->OutPutCamera(aNameIm);
       cCameraTiepRed * aCam = new cCameraTiepRed(*this,aNameIm,aCS);
       
       // Put them in vector and map
       mVecCam.push_back(aCam);
       mMapCam[aNameIm] = aCam;

       // get box of footprint and resolution
       Box2dr aBox = aCam->CS().BoxSol();
       aPInf = Inf(aBox._p0,aPInf);
       aPSup = Sup(aBox._p1,aPSup);
       aVResol.push_back(aCam->CS().ResolutionSol());
       // std::cout << "BBB " << aBox._p0 << aBox._p1 << " " <<  aCam->CS().ResolutionSol() << "\n";
   }
   // Memorize the global box
   mBoxGlob = Box2dr(aPInf,aPSup);
   // Get a global resolution as mediane of each resolution
   mResol = MedianeSup(aVResol);

   std::cout << "   BOX " << mBoxGlob << " Resol=" << mResol << "\n";
}


const std::string cAppliTiepRed::TheNameTmp = "Tmp-ReducTieP/";

std::string  cAppliTiepRed::NameParamBox(int aK,bool Bin) const
{
    return mDir+TheNameTmp + "Param_" +ToString(aK) + (Bin ? ".xml" : "dmp");
}

std::string  cAppliTiepRed::DirOneImage(const std::string &aName) const
{
   return mDir+TheNameTmp + aName + "/";
}

std::string  cAppliTiepRed::NameHomol(const std::string &aName1,const std::string &aName2,int aK) const
{
   return DirOneImage(aName1) + "KBOX" + ToString(aK) + "-" + aName2  + ".dat";
}




cVirtInterf_NewO_NameManager & cAppliTiepRed::NM(){ return *mNM ;}
const cXml_ParamBoxReducTieP & cAppliTiepRed::ParamBox() const {return mXmlParBox;}
const double & cAppliTiepRed::ThresoldPrec2Point() const {return  mPrec2Point;}
const int    & cAppliTiepRed::ThresholdNbPts2Im() const  {return mThresholdNbPts2Im;}
const int    & cAppliTiepRed::ThresholdTotalNbPts2Im() const  {return mThresholdTotalNbPts2Im;}
cCameraTiepRed * cAppliTiepRed::KthCam(int aK) {return mVecCam[aK];}
const double & cAppliTiepRed::ThresholdPrecMult() const {return mThresholdPrecMult;}
const double & cAppliTiepRed::StdPrec() const {return mStdPrec;}
std::vector<int>  & cAppliTiepRed::BufICam() {return mBufICam;}
cInterfChantierNameManipulateur* cAppliTiepRed::ICNM() {return mICNM;}




void cAppliTiepRed::AddLnk(cLnk2ImTiepRed * aLnk)
{
    mLnk2Im.push_back(aLnk);
}



void TestPoly(const cElPolygone & aPol )
{
   const std::list<std::vector<Pt2dr> >   aLC = aPol.Contours();

   for
   (
       std::list<std::vector<Pt2dr> >::const_iterator itC=aLC.begin();
       itC!=aLC.end();
       itC++
   )
   {
        std::cout << " AAaa= " << itC->size()  ;
        for (int aK=0 ; aK<int(itC->size()) ; aK++)
            std::cout << (*itC)[aK] << "  ";
        std::cout << "\n";
   }
   
   std::cout << "\n";
}


void cAppliTiepRed::GenerateSplit()
{
    ELISE_fp::MkDirSvp(mDir+TheNameTmp);
    for (int aKI = 0 ; aKI<int(mFilesIm->size()) ; aKI++)
    {
       const std::string & aNameIm = (*mFilesIm)[aKI];
       ELISE_fp::MkDirSvp(DirOneImage(aNameIm));
    }

    Pt2dr aSzPix = mBoxGlob.sz() / mResol; // mBoxGlob.sz()  mResol => local refernce,  aSzPix => in pixel (average)
    Pt2di aNb = round_up(aSzPix / double(mSzTile));  // Compute the number of boxes
    std::cout << "   GenerateSplit SzP=" << aSzPix << " Nb=" << aNb << " \n";
    Pt2dr aSzTile =  mBoxGlob.sz().dcbyc(Pt2dr(aNb));  // Compute the size of each tile 

    std::list<std::string> aLCom;


    int aCpt=0;
    // Parse the tiles
    for (int aKx=0 ; aKx<aNb.x ; aKx++)
    {
        for (int aKy=0 ; aKy<aNb.y ; aKy++)
        {
             Pt2dr aP0 = mBoxGlob._p0 + aSzTile.mcbyc(Pt2dr(aKx,aKy)); // Origine of tile
             Pt2dr aP1 = aP0 +aSzTile;                                 // End of tile
             Box2dr aBox(aP0,aP1);                                     // Box of tile
             cElPolygone aPolyBox = cElPolygone::FromBox(aBox);         // Box again in polygon
             // std::cout << "JJJJJjj " << aP0 << aP1 << "\n";
             cXml_ParamBoxReducTieP aParamBox;                           // XML/C++ Structure to save
             aParamBox.Box() = aBox;                                     // Memorize the box of tile

             std::vector<cCameraTiepRed *> aVCamSel;
             for (int aKC=0 ; aKC<int(mVecCam.size()) ; aKC++)
             {
                   // Intersection between footprint and box (see class cElPolygone)
                   cElPolygone  aPolInter = aPolyBox  * mVecCam[aKC]->CS().EmpriseSol(); 
                   // If polygon not empty
                   if (aPolInter.Surf() > 0)
                   {
                        //  Add the name to the vector
                        aParamBox.Ims().push_back(mVecCam[aKC]->NameIm());
                        aVCamSel.push_back(mVecCam[aKC]);
                   }

             }
              // If at least 2 images
             if (aParamBox.Ims().size() >=2)
             {
                 // Save the file to XML
                 MakeFileXML(aParamBox,NameParamBox(aCpt,false));
                 MakeFileXML(aParamBox,NameParamBox(aCpt,true));
                 // Generate the command line to process this box
                 std::string aCom = GlobArcArgv + "  KBox=" + ToString(aCpt);
                 // add to list to be executed
                 aLCom.push_back(aCom);
                 std::cout << "==>   " << aCom << "\n";
                 for (int aKC1=0; aKC1<int(aVCamSel.size()) ; aKC1++)
                 {
                     cCameraTiepRed * aCam1 = aVCamSel[aKC1];
                     for (int aKC2=0; aKC2<int(aVCamSel.size()) ; aKC2++)
                     {
                         cCameraTiepRed * aCam2 = aVCamSel[aKC2];
                         cElPolygone  aPolInter = aPolyBox  * aCam1->CS().EmpriseSol() *  aCam2->CS().EmpriseSol();
                         if (aPolInter.Surf() > 0)
                         {
                            aCam1->AddCamBox(aCam2,aCpt);
                         }
                     }
                 }
                 aCpt++;

             }
        }
    }
    // cEl_GPAO::DoComInParal(aLCom);
    cEl_GPAO::DoComInSerie(aLCom);

    for (int aKC=0 ; aKC<int(mVecCam.size()) ; aKC++)
       mVecCam[aKC]->SaveHom();
}



void  cAppliTiepRed::Exe()
{
   if (mCallBack)
   {
      DoReduceBox();
   }
   else
   {
       GenerateSplit();
   }
}



int TestOscarTieP_main(int argc,char **argv) 
{
    cAppliTiepRed * anAppli = new cAppliTiepRed(argc,argv);

     anAppli->Exe();


    return EXIT_SUCCESS;
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
