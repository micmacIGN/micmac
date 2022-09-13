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

#include "TiePHistorical.h"



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


void EnhacedSpG(std::string aImg1, std::string aImg2, Pt2dr aPatchLSz, Pt2dr aBufferLSz, Pt2dr aPatchRSz, Pt2dr aBufferRSz, int nRotate, cCommonAppliTiepHistorical aCAS3D, bool aExe, double aCheckNb)
{
    std::string aOutDir = "./Tmp_Patches-CoReg";

    /**************************************/
    /* 1.2 - GetPatchPair for rough co-registration */
    /**************************************/
    std::string aCom = "";
    //if (!EAMIsInit(&aCAS3D.mOutDir))   aCom +=  " OutDir=" + aOutDir;
    aCom += " PatchLSz=[" + ToString(aPatchLSz.x) + "," + ToString(aPatchLSz.y) + "]";
    aCom += " BufferLSz=[" + ToString(aBufferLSz.x) + "," + ToString(aBufferLSz.y) + "]";
    aCom += " PatchRSz=[" + ToString(aPatchRSz.x) + "," + ToString(aPatchRSz.y) + "]";
    aCom += " BufferRSz=[" + ToString(aBufferRSz.x) + "," + ToString(aBufferRSz.y) + "]";
    std::string aFullCom = StdCom("TestLib GetPatchPair BruteForce", aImg1 + BLANK + aImg2 + BLANK + aCom + BLANK + "Rotate=" + ToString(nRotate) + BLANK + aCAS3D.ComParamGetPatchPair(), aExe);
    //cout<<aFullCom<<endl;
// + ToString(nRotate)
//    std::string aImg1 = aCAS3D.GetFolderName(mDSMDirL) + "." + StdPostfix(aDSMImgNameL);
//    std::string aImg2 = aCAS3D.GetFolderName(mDSMDirR) + "." + StdPostfix(aDSMImgNameR);

    std::string aRotate[4] = {"", "_R90", "_R180", "_R270"};
    std::string aFinalOutSH;
    int nMaxinlier = 0;
    //Rotate the master DSM 4 times and apply superGlue
    for(int i=0; i<4; i++)
    {
        if(nRotate != -1)
        {
            std::string aRotateDSMStr = "_R" + ToString(nRotate);
            if(nRotate == 0)
                aRotateDSMStr = "";
            if(aRotate[i] != aRotateDSMStr)
            {
                printf("%dth attempt with \"%s\" doesn't match with \"%s\", hence skipped\n", i, aRotate[i].c_str(), aRotateDSMStr.c_str());
                continue;
            }
        }

        /**************************************/
        /* 1.3 - SuperGlue for rough co-registration */
        /**************************************/
        std::string aImgPair = StdPrefix(aCAS3D.mImgPair) + aRotate[i] + "." + StdPostfix(aCAS3D.mImgPair);
        cout<<"aImgPair: "<<aImgPair<<endl;
        aCom = "";
        if (!EAMIsInit(&aCAS3D.mInput_dir))    aCom +=  " InDir=" + aOutDir+"/";
        //if (!EAMIsInit(&aCAS3D.mOutput_dir))   aCom +=  " SpGOutDir=" + aOutDir+"/";
        aCom +=  " CheckNb=\" " + ToString(aCheckNb) + "\"";
        aCom +=  " Rotate=" + ToString(nRotate);
        aCom += " Resize=[" + ToString(aCAS3D.mResize.x) + "," + ToString(aCAS3D.mResize.y) + "]";
        aCom += " Viz=" + ToString(aCAS3D.mViz);
        aFullCom = StdCom("TestLib SuperGlue", aImgPair + BLANK + aCom + BLANK + aCAS3D.ComParamSuperGlue(), aExe);
        //cout<<aFullCom<<endl;


        /**************************************/
        /* 1.4 - MergeTiePt for rough co-registration */
        /**************************************/
        std::string aHomoXml = StdPrefix(aCAS3D.mHomoXml) + aRotate[i] + "." + StdPostfix(aCAS3D.mHomoXml);
        aCom = "";
        if (!EAMIsInit(&aCAS3D.mHomoXml))   aCom +=  " HomoXml=" + aHomoXml;
        if (!EAMIsInit(&aCAS3D.mMergeTiePtInSH))   aCom +=  " MergeInSH=" + aCAS3D.mSpGlueOutSH;
        aCom +=  " PatchSz=[" + ToString(aPatchLSz.x) + "," + ToString(aPatchLSz.y) + "]";
        aCom +=  " BufferSz=[" + ToString(aBufferLSz.x) + "," + ToString(aBufferLSz.y) + "]";
        aFullCom = StdCom("TestLib MergeTiePt", aOutDir+"/" + BLANK + aCom + BLANK + aCAS3D.ComParamMergeTiePt(), aExe);
        //cout<<aFullCom<<endl;


        /**************************************/
        /* 1.5 - RANSAC R2D for rough co-registration */
        /**************************************/
        aCom = "";
        if (!EAMIsInit(&aCAS3D.mR2DInSH))   aCom +=  " 2DRANInSH=-" + StdPrefix(aHomoXml);
        std::string aRANSACOutSH = "-" + StdPrefix(aHomoXml) + "-2DRANSAC";
        aFullCom = StdCom("TestLib RANSAC R2D", aImg1 + BLANK + aImg2 + BLANK + "Dir=" + aOutDir+"/" + BLANK + aCom + BLANK + aCAS3D.ComParamRANSAC2D(), aExe);
        //cout<<aFullCom<<endl;
        int nInlier = GetTiePtNum(aOutDir, aImg1, aImg2, aRANSACOutSH);
        cout<<i<<",,"<<aRANSACOutSH<<","<<nInlier<<endl;

        if(nInlier > nMaxinlier)
        {
            nMaxinlier = nInlier;
            aFinalOutSH = aRANSACOutSH;
        }
    }
    cout<<"aFinalOutSH: "<<aFinalOutSH<<endl;
    cout<<"Final correspondences are saved in "<<aOutDir<<"/Homol"<<aFinalOutSH<<endl;
}

int EnhancedSpG_main(int argc,char ** argv)
{
   cCommonAppliTiepHistorical aCAS3D;

   std::string aImgList1;
   std::string aImgList2;

   std::string aImg1;
   std::string aImg2;

   double aCheckNb=-1;
   int aRotate=-1;
//   Pt2dr aPatchSz = Pt2dr(640, 480);
//   Pt2dr aBufferSz = Pt2dr(0, 0);
   Pt2dr aPatchLSz(640, 480);
   Pt2dr aBufferLSz(0,0);
   Pt2dr aPatchRSz(640, 480);
   Pt2dr aBufferRSz(0,0);

   bool aExe=true;

   /*
   std::string aSubPatchXml = "SubPatch.xml";
   std::string aPatchDir = "./Tmp_Patches";

   Pt2dr aPatchSz(640, 480);
   Pt2dr aBufferSz(-1, -1);
*/

   ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aImgList1,"ImgList1: All RGB images in epoch1 (Dir+Pattern, or txt file of image list)")
               << EAMC(aImgList2,"ImgList2: All RGB images in epoch2 (Dir+Pattern, or txt file of image list)"),
        LArgMain()
               << EAM(aExe,"Exe",true,"Execute all, Def=true. If this parameter is set to false, the pipeline will not be executed and the command of all the submodules will be printed.")
                    << aCAS3D.ArgBasic()
               << aCAS3D.ArgGetPatchPair()
               << aCAS3D.ArgSuperGlue()
               << aCAS3D.ArgMergeTiePt()
               << aCAS3D.Arg2DRANSAC()
               << EAM(aCheckNb,"CheckNb",true,"Radius of the search space for SuperGlue (which means correspondence [(xL, yL), (xR, yR)] with (xL-xR)*(xL-xR)+(yL-yR)*(yL-yR) > CheckNb*CheckNb will be removed afterwards), Def=-1 (means don't check search space)")
               << EAM(aRotate,"Rotate",true,"The angle of clockwise rotation from the master image to the secondary image (only 4 options available: 0, 90, 180, 270, as SuperGlue is invariant to rotation smaller than 45 degree.), Def=-1 (means all the 4 options will be executed, and the one with the most inlier will be kept) ")
//               << EAM(aPatchSz, "PatchSz", true, "Patch size of the tiling scheme, which means the images to be matched by SuperGlue will be split into patches of this size, Def=[640,480]")
//               << EAM(aBufferSz, "BufferSz", true, "Buffer zone size around the patch of the tiling scheme, Def=[0,0]")

               << EAM(aPatchLSz, "PatchLSz", true, "Patch size of the tiling scheme for master image, which means the master image to be matched by SuperGlue will be split into patches of this size, Def=[640, 480]")
               << EAM(aBufferLSz, "BufferLSz", true, "Buffer zone size around the patch of the tiling scheme for master image, Def=[0,0]")
               << EAM(aPatchRSz, "PatchRSz", true, "Patch size of the tiling scheme for secondary image, which means the secondary image to be matched by SuperGlue will be split into patches of this size, Def=[640, 480]")
               << EAM(aBufferRSz, "BufferRSz", true, "Buffer zone size around the patch of the tiling scheme for secondary image, Def=[0,0]")

               );

   std::vector<std::string> aVIm1;
   std::vector<std::string> aVIm2;
   GetImgListVec(aImgList1, aVIm1);
   GetImgListVec(aImgList2, aVIm2);

   for(int i=0; i<int(aVIm1.size()); i++)
   {
       aImg1 = aVIm1[i];
       for(int j=0; j<int(aVIm2.size()); j++)
       {
           aImg2 = aVIm2[j];
           EnhacedSpG(aImg1, aImg2, aPatchLSz, aBufferLSz, aPatchRSz, aBufferRSz, aRotate, aCAS3D, aExe, aCheckNb);
       }
   }

   return EXIT_SUCCESS;
}
