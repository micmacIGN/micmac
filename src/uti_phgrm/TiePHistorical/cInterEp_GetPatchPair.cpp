//#define aNbType 2
//std::string  Type[aNbType] = {"BruteForce","Guided"};


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

bool FallInBox(Pt2dr* aPCorner, Pt2dr aLeftTop, Pt2di aRightLower)
{
    if((aPCorner[0].x < aLeftTop.x) && (aPCorner[1].x < aLeftTop.x) && (aPCorner[2].x < aLeftTop.x) && (aPCorner[3].x < aLeftTop.x))
        return false;

    if((aPCorner[0].y < aLeftTop.y) && (aPCorner[1].y < aLeftTop.y) && (aPCorner[2].y < aLeftTop.y) && (aPCorner[3].y < aLeftTop.y))
        return false;

    if((aPCorner[0].x > aRightLower.x) && (aPCorner[1].x > aRightLower.x) && (aPCorner[2].x > aRightLower.x) && (aPCorner[3].x > aRightLower.x))
        return false;

    if((aPCorner[0].y > aRightLower.y) && (aPCorner[1].y > aRightLower.y) && (aPCorner[2].y > aRightLower.y) && (aPCorner[3].y > aRightLower.y))
        return false;

    return true;
}

std::string GetFileName(std::string strIn)
{
    std::string strOut = strIn;

    std::size_t found = strIn.find("/");
    if (found!=std::string::npos)
        strOut = strIn.substr(found+1, strIn.length());

    return strOut;
}

std::string GetFolderName(std::string strIn)
{
    std::string strOut = strIn;

    std::size_t found = strIn.find("/");
    if (found!=std::string::npos)
        strOut = strIn.substr(0, found);

    return strOut;
}

void WriteXml(std::string aImg1, std::string aImg2, std::string aSubPatchXml, std::vector<std::string> vPatchesL, std::vector<std::string> vPatchesR, std::vector<cElHomographie> vHomoL, std::vector<cElHomographie> vHomoR)
{
    //cout<<aSubPatchXml<<endl;
    cSetOfPatches aDAFout;

    cMes1Im aIms1;
    cMes1Im aIms2;

    aIms1.NameIm() = GetFileName(aImg1);
    aIms2.NameIm() = GetFileName(aImg2);

    //cout<<aIms1.NameIm()<<endl;

    int nPatchNumL = vPatchesL.size();
    int nPatchNumR = vPatchesR.size();
    for(int i=0; i < nPatchNumL; i++)
    {
        //cout<<i<<"/"<<nPatchNum<<endl;
        cOnePatch1I patch1;
        patch1.NamePatch() = vPatchesL[i];
        patch1.PatchH() = vHomoL[i].ToXml();
        aIms1.OnePatch1I().push_back(patch1);
    }

    for(int i=0; i < nPatchNumR; i++)
    {
        cOnePatch1I patch2;
        patch2.NamePatch() = vPatchesR[i];
        patch2.PatchH() = vHomoR[i].ToXml();
        aIms2.OnePatch1I().push_back(patch2);
    }

    aDAFout.Mes1Im().push_back(aIms1);
    aDAFout.Mes1Im().push_back(aIms2);

    MakeFileXML(aDAFout, aSubPatchXml);
}

//simply clip images to get left patches (m patches), and resample images to get right patches (m patches). The number of pairs to be matched will be m.
//mainly used for precise matching
void GetPatchPair(std::string aOutDir, std::string aOutImg1, std::string aOutImg2, std::string aImg1, std::string aImg2, std::string aIm1OriFile, std::string aIm2OriFile, Pt2dr aPatchSz, Pt2dr aBufferSz, std::string aImgPair, std::string aDir, std::string aSubPatchXml, cTransform3DHelmert aTrans3DH, std::string aDSMFileL, std::string aDSMDirL, bool bPrint=false, std::string aPrefix="")
{
    std::string aOriginImg1 = aImg1;

    aImg1 = aPrefix + aImg1;
    aImg2 = aPrefix + aImg2;
    aOutImg1 = aPrefix + aOutImg1;
    aOutImg2 = aPrefix + aOutImg2;

    if (ELISE_fp::exist_file(aDir+aImg1) == false || ELISE_fp::exist_file(aDir+aImg2) == false)
    {
        cout<<aDir+aImg1<<" or "<<aDir+aImg2<<" didn't exist, hence skipped"<<endl;
        return;
    }

    Tiff_Im aRGBIm1((aDir+aImg1).c_str());
    Pt2di ImgSzL = aRGBIm1.sz();
    Tiff_Im aRGBIm2((aDir+aImg2).c_str());
    Pt2di ImgSzR = aRGBIm2.sz();

    Pt2dr CoreaPatchSz;
    CoreaPatchSz.x = aPatchSz.x - aBufferSz.x*2;
    CoreaPatchSz.y = aPatchSz.y - aBufferSz.y*2;

    Pt2di PatchNum;
    PatchNum.x = ceil(ImgSzL.x*1.0/CoreaPatchSz.x);
    PatchNum.y = ceil(ImgSzL.y*1.0/CoreaPatchSz.y);


    //cInterfChantierNameManipulateur * anICNM = cInterfChantierNameManipulateur::BasicAlloc("./");


    //ElCamera * aCamL = CamOrientGenFromFile(aNameOriL+"/Orientation-"+aImg1+".xml", anICNM);
    //ElCamera * aCamR = CamOrientGenFromFile(aNameOriR+"/Orientation-"+aImg2+".xml", anICNM);

    //ElCamera * aCamL = BasicCamOrientGenFromFile(aIm1OriFile);
    ElCamera * aCamR = BasicCamOrientGenFromFile(aIm2OriFile);


    ElCamera * aCamL = BasicCamOrientGenFromFile(aIm1OriFile);
    double prof_d = aCamL->GetProfondeur();
    double altiSol = aCamL->GetAltiSol();
    if(bPrint)
    {
        printf("GetProfondeur: %.2lf, GetAltiSol: %.2lf\n", prof_d, altiSol);
    }


    Pt2dr aPCornerPatch[4];
    Pt2dr origin = Pt2dr(0, 0);
    aPCornerPatch[0] = origin;
    aPCornerPatch[1] = Pt2dr(origin.x+aPatchSz.x, origin.y);
    aPCornerPatch[2] = Pt2dr(origin.x+aPatchSz.x, origin.y+aPatchSz.y);
    aPCornerPatch[3] = Pt2dr(origin.x, origin.y+aPatchSz.y);

    //save the name and homography of patches, for saving the xml file latter
    std::vector<std::string> vPatchesL, vPatchesR;
    std::vector<cElHomographie> vHomoL, vHomoR;

    std::vector<string> vaImgPair;
    int m, n;
    std::list<std::string> aLComClip, aLComResample;
    std::string aComBaseClip = MMBinFile(MM3DStr) + "ClipIm " + aImg1 + " ";
    std::string aComBaseResample = MMBinFile(MM3DStr) + "TestLib OneReechFromAscii ";
    std::string aClipSz = " ["+std::to_string(int(aPatchSz.x))+","+std::to_string(int(aPatchSz.y))+"] ";

    //Because TestLib OneReechFromAscii output the result in the same directory of aImg2, so we need to move the result txt and tif file into aOutDir
    std::list<std::string> aComMv;

    cout<<"Patch number of "<<aOutImg1<<": "<<PatchNum.x<<"*"<<PatchNum.y<<"="<<PatchNum.x*PatchNum.y<<endl;

    for(m=0; m<PatchNum.x; m++)
    {
        for(n=0; n<PatchNum.y; n++)
        {
            std::string aSubImg1 = StdPrefix(aOutImg1) + "_" + std::to_string(m) + "_" + std::to_string(n) + "." + StdPostfix(aOutImg1);

            origin.x = m*CoreaPatchSz.x - aBufferSz.x;
            origin.y = n*CoreaPatchSz.y - aBufferSz.y;

            // 1. use ClipIm command to clip first image into patches
            std::string aComClipFirstImg = aComBaseClip + " ["+std::to_string(int(origin.x))+","+std::to_string(int(origin.y))+"] " + aClipSz + " Out="+aOutDir+"/"+aSubImg1;
            cout<<aComClipFirstImg<<endl;
            aLComClip.push_back(aComClipFirstImg);

            cElComposHomographie aFstHX(1, 0, origin.x);
            cElComposHomographie aFstHY(0, 1, origin.y);
            cElComposHomographie aFstHZ(0, 0,        1);
            cElHomographie  aFstH =  cElHomographie(aFstHX,aFstHY,aFstHZ);

            // 2. use "TestLib OneReechFromAscii" command to clip second image into patches by reprojecting the first patches to the second image
            if(true)
            {
                double dScaleL = 1;
                double dScaleR = 1;

                Pt2dr aPCornerL[4];
                aPCornerL[0] = origin;
                aPCornerL[1] = Pt2dr(origin.x+aPatchSz.x, origin.y);
                aPCornerL[2] = Pt2dr(origin.x+aPatchSz.x, origin.y+aPatchSz.y);
                aPCornerL[3] = Pt2dr(origin.x, origin.y+aPatchSz.y);

                Pt2dr aPCornerR[4];
                for(int i=0; i<4; i++)
                {
                    Pt2dr aP1 = aPCornerL[i];
                    aP1.x = aP1.x*dScaleL;
                    aP1.y = aP1.y*dScaleL;

                    Pt3dr aPTer1;
                    if(aDSMDirL.length() > 0)
                    {
                        cGet3Dcoor a3DCoorL(aIm1OriFile);
                        TIm2D<float,double> aTImProfPxL = a3DCoorL.SetDSMInfo(aDSMFileL, aDSMDirL);
                        bool bValidL;
                        aPTer1 = a3DCoorL.Get3Dcoor(aP1, aTImProfPxL, bValidL);
                    }
                    else
                        aPTer1 = aCamL->ImEtProf2Terrain(aP1, prof_d);
                    //Pt3dr Pt_H_sol = aCamL->ImEtZ2Terrain(aP1, altiSol);

                    aPTer1 = aTrans3DH.Transform3Dcoor(aPTer1);

                    aPCornerR[i] = aCamR->R3toF2(aPTer1);
                    aPCornerR[i].x = aPCornerR[i].x/dScaleL/dScaleR;
                    aPCornerR[i].y = aPCornerR[i].y/dScaleL/dScaleR;

                    if(bPrint)
                    {
                        printf("%dth: CornerL: [%.2lf\t%.2lf], ImEtProf2Terrain: [%.2lf\t%.2lf\t%.2lf], CornerR: [%.2lf\t%.2lf]\n", i, aPCornerL[i].x, aPCornerL[i].y, aPTer1.x, aPTer1.y, aPTer1.z, aPCornerR[i].x, aPCornerR[i].y);
                        //printf("ImEtZ2Terrain: [%.2lf\t%.2lf\t%.2lf]\n", Pt_H_sol.x, Pt_H_sol.y, Pt_H_sol.z);
                    }
                }

                std::string aNameSave = StdPrefix(aOutImg2) + "_" + StdPrefix(aOriginImg1) + "_" + std::to_string(m) + "_" + std::to_string(n);
                std::string aSubImg2 = aNameSave + "." + StdPostfix(aOutImg2);
                aNameSave += ".txt";
                //cout<<aNameSave<<endl;
                if(FallInBox(aPCornerR, Pt2dr(0,0), ImgSzR) == true)
                {
                    vaImgPair.push_back(aSubImg1 + " " + aSubImg2);
                    //cout<<"fall in box"<<endl;
                    FILE * fpOutput = fopen((aNameSave).c_str(), "w");
                    for(int i=0; i<4; i++)
                    {
                        fprintf(fpOutput, "%lf %lf %lf %lf\n", aPCornerPatch[i].x, aPCornerPatch[i].y, aPCornerR[i].x, aPCornerR[i].y);
                    }
                    fclose(fpOutput);

                    std::string aComResampleSndImg = aComBaseResample + aImg2  + " " + aNameSave + " Out="+aSubImg2 + " Show=true";
                    cout<<aComResampleSndImg<<endl;
                    aLComResample.push_back(aComResampleSndImg);

                    std::string aMvTxt = "mv "+aNameSave + " "+aOutDir+"/"+aNameSave;
                    std::string aMvTif = "mv "+aSubImg2 + " "+aOutDir+"/"+aSubImg2;
                    aComMv.push_back(aMvTxt);
                    aComMv.push_back(aMvTif);
                }
                else
                {
                    if(bPrint)
                        cout<<aNameSave<<" out of border, hence the current patch is not saved"<<endl;
                }

                    //Save the homography, this is copied from function "cAppliReechHomogr::cAppliReechHomogr(int argc,char ** argv)  :" in src/uti_phgrm/CPP_CreateEpip.cpp, where the patch is resampled
                    ElPackHomologue aPack;
                    //aPack = ElPackHomologue::FromFile(aNameSave);
                    for(int i=0; i<4; i++)
                    {
                        aPack.Cple_Add(ElCplePtsHomologues(aPCornerPatch[i], aPCornerR[i]));
                    }

                    double anEcart,aQuality;
                    bool Ok;
                    cElHomographie aSndH = cElHomographie::RobustInit(anEcart,&aQuality,aPack,Ok,50,80.0,2000);

                    vPatchesL.push_back(aSubImg1);
                    vHomoL.push_back(aFstH);
                    vPatchesR.push_back(aSubImg2);
                    vHomoR.push_back(aSndH);
            }
            //end
        }
    }

    //write SuperGlueInput.txt
    //std::string aDir = "/home/lulin/Documents/zll/TestLulinCodeInMicMac/SpGlue/new/";
    FILE * fpOutput = fopen((aOutDir+aImgPair).c_str(), "w");
    int nPairNum = vaImgPair.size();
    for(int i=0; i<nPairNum; i++)
    {
        fprintf(fpOutput, "%s", vaImgPair[i].c_str());
        if(i<nPairNum-1)
            fprintf(fpOutput, "\n");
    }
    fclose(fpOutput);

    WriteXml(aImg1, aImg2, aOutDir+aSubPatchXml, vPatchesL, vPatchesR, vHomoL, vHomoR);

    cEl_GPAO::DoComInParal(aLComClip);
    cEl_GPAO::DoComInSerie(aLComResample);
    cEl_GPAO::DoComInParal(aComMv);

    /* if use cEl_GPAO::DoComInParal(aLComResample), something like this will happen:
    Error while file reading |
    FILE = ./Tmp-MM-Dir/14FRPCAB35x00014_01036.tif_Ch1.tif  pos = 20971528|
 reading 1 , got 0|------------------------------------------------------------
|   Sorry, the following FATAL ERROR happened
|
|    Error while file reading
|
------------------------------------------------------------
-------------------------------------------------------------
|       (Elise's)  LOCATION :
|
| Error was detected
|          at line : 1255
|          of file : /home/lulin/micmac/src/util/files.cpp
-------------------------------------------------------------
*/
}

Pt2dr ClipImg(std::string aOutImg1, std::string aImg1, Pt2di ImgSzL, Pt2dr aPatchSz, Pt2dr aBufferSz, Pt2dr origin, std::string aOutDir, std::list<std::string>& aLComClip, std::vector<std::string>& vPatchesL, std::vector<cElHomographie>& vHomoL)
{
    Pt2dr CoreaPatchSz;
    CoreaPatchSz.x = aPatchSz.x - aBufferSz.x*2;
    CoreaPatchSz.y = aPatchSz.y - aBufferSz.y*2;

    Pt2dr PatchNum;
    PatchNum.x = int(ImgSzL.x*1.0/CoreaPatchSz.x)+1;
    PatchNum.y = int(ImgSzL.y*1.0/CoreaPatchSz.y)+1;

    std::string aComBaseClip = MMBinFile(MM3DStr) + "ClipIm " + aImg1 + " ";
    std::string aClipSz = " ["+std::to_string(int(aPatchSz.x))+","+std::to_string(int(aPatchSz.y))+"] ";

    cout<<"Patch number of "<<aOutImg1<<": "<<PatchNum.x<<"*"<<PatchNum.y<<"="<<PatchNum.x*PatchNum.y<<endl;

    int m, n;
    for(m=0; m<PatchNum.x; m++)
    {
        for(n=0; n<PatchNum.y; n++)
        {
            std::string aSubImg1 = StdPrefix(aOutImg1) + "_" + std::to_string(m) + "_" + std::to_string(n) + "." + StdPostfix(aOutImg1);

            origin.x = m*CoreaPatchSz.x - aBufferSz.x;
            origin.y = n*CoreaPatchSz.y - aBufferSz.y;

            // 1. use ClipIm command to clip first image into patches
            std::string aComClipFirstImg = aComBaseClip + " ["+std::to_string(int(origin.x))+","+std::to_string(int(origin.y))+"] " + aClipSz + "Out="+aOutDir+"/"+aSubImg1;
            //cout<<aComClipFirstImg<<endl;
            aLComClip.push_back(aComClipFirstImg);

            cElComposHomographie aFstHX(1, 0, origin.x);
            cElComposHomographie aFstHY(0, 1, origin.y);
            cElComposHomographie aFstHZ(0, 0,        1);
            cElHomographie  aFstH =  cElHomographie(aFstHX,aFstHY,aFstHZ);

            vPatchesL.push_back(aSubImg1);
            vHomoL.push_back(aFstH);
        }
    }
    return PatchNum;
}

//simply clip images to get left patches (m patches) and right patches (n patches).  The number of pairs to be matched will be m*n.
//mainly used for rough co-registration
//aOriImg1 is the orginal left image, aImg1 could be the same as aOriImg1 (in this case aIm1_OriImg1 is unit matrix), or rotated image based on aOriImg1
void GetTilePair(std::string aOutDir, std::string aOriOutImg1, std::string aRotateOutImg1, std::string aOutImg2, std::string aImg1, std::string aImg2, Pt2dr aPatchSz, Pt2dr aBufferSz, std::string aImgPair, std::string aDir, std::string aSubPatchXml, std::string aOriImg1, cElHomographie aIm1_OriImg1)
{
    //cout<<aDir<<endl;
    if (ELISE_fp::exist_file(aDir+aImg1) == false || ELISE_fp::exist_file(aDir+aImg2) == false)
    {
        cout<<aDir+aImg1<<" or "<<aDir+aImg2<<" didn't exist, hence skipped"<<endl;
        return;
    }

    std::string strCpImg;
    strCpImg = "cp "+aDir+aOriImg1+" "+aOutDir+aOriOutImg1;
    cout<<strCpImg<<endl;
    System(strCpImg);
    strCpImg = "cp "+aDir+aImg2+" "+aOutDir+aOutImg2;
    cout<<strCpImg<<endl;
    System(strCpImg);

    Tiff_Im aDSMIm1((aDir+aImg1).c_str());
    Pt2di ImgSzL = aDSMIm1.sz();
    Tiff_Im aDSMIm2((aDir+aImg2).c_str());
    Pt2di ImgSzR = aDSMIm2.sz();

    Pt2dr origin = Pt2dr(0, 0);

    std::vector<std::string> vPatchesL, vPatchesR;
    std::vector<cElHomographie> vHomoL, vHomoR;

    std::list<std::string> aLComClip, aRComClip;

    Pt2dr aPatchNumL = ClipImg(aRotateOutImg1, aImg1, ImgSzL, aPatchSz, aBufferSz, origin, aOutDir, aLComClip, vPatchesL, vHomoL);
    Pt2dr aPatchNumR = ClipImg(aOutImg2, aImg2, ImgSzR, aPatchSz, aBufferSz, origin, aOutDir, aRComClip, vPatchesR, vHomoR);

    cout<<"Number of tile pairs: "<<aPatchNumL.x*aPatchNumL.y*aPatchNumR.x*aPatchNumR.y<<endl;

    std::vector<string> vaImgPair;
    for(unsigned int i=0; i<vPatchesL.size(); i++)
    {
        for(unsigned int j=0; j<vPatchesR.size(); j++)
        {
            vaImgPair.push_back(vPatchesL[i] + " " + vPatchesR[j]);
        }
    }

    //write SuperGlueInput.txt
    //std::string aDir = "/home/lulin/Documents/zll/TestLulinCodeInMicMac/SpGlue/new/";
    FILE * fpOutput = fopen((aOutDir+aImgPair).c_str(), "w");
    int nPairNum = vaImgPair.size();
    for(int i=0; i<nPairNum; i++)
    {
        fprintf(fpOutput, "%s", vaImgPair[i].c_str());
        if(i<nPairNum-1)
            fprintf(fpOutput, "\n");
    }
    fclose(fpOutput);

    //transform the homography
    int nHomoLNum = vHomoL.size();
    for(int i=0; i<nHomoLNum; i++)
    {
        vHomoL[i] = aIm1_OriImg1*vHomoL[i];
    }

    WriteXml(aOriOutImg1, aOutImg2, aOutDir+aSubPatchXml, vPatchesL, vPatchesR, vHomoL, vHomoR);

    cEl_GPAO::DoComInParal(aLComClip);
    cEl_GPAO::DoComInParal(aRComClip);
}


void RotateImgBy90Deg(std::string aDir, std::string aImg1, std::string aNameOut)
{
    Tiff_Im aDSMIm1((aDir+"/"+aImg1).c_str());
    Pt2di ImgSzL = aDSMIm1.sz();

    aNameOut = aDir+"/"+aNameOut;

    //std::string aNameOut = aDir+"/"+StdPrefix(aImg1)+"_R90.tif";

    Tiff_Im TiffOut  =     Tiff_Im
                           (
                              aNameOut.c_str(),
                              Pt2di(ImgSzL.y, ImgSzL.x),
                              aDSMIm1.type_el(),
                              Tiff_Im::No_Compr,
                              aDSMIm1.phot_interp(),
                              Tiff_Im::Empty_ARG
                          );

    TIm2D<float, double> aTImProfPx(ImgSzL);
    ELISE_COPY
    (
    aTImProfPx.all_pts(),
    aDSMIm1.in(),
    aTImProfPx.out()
    );

    TIm2D<float, double> aTImProfPxTmp(Pt2di(ImgSzL.y, ImgSzL.x));
    ELISE_COPY
    (
    aTImProfPxTmp.all_pts(),
    aTImProfPx.in()[Virgule(FY,FX)],
    aTImProfPxTmp.out()
    );

    //flip
    ELISE_COPY
    (
    TiffOut.all_pts(),
    aTImProfPxTmp.in(0)[Virgule(ImgSzL.y-FX,FY)],
    TiffOut.out()
    );
}

void RotateImgBy90DegNTimes(std::string aDir, std::string aImg1, std::string aNameOut, int nTime)
{
    RotateImgBy90Deg(aDir, aImg1, aNameOut);
    nTime--;

    for(int i=0; i<nTime; i++)
    {
        std::string aNameOutTmp = StdPrefix(aNameOut)+"_tmp."+StdPostfix(aNameOut);
        std::string strCpImg = "mv "+aDir+aNameOut+" "+aDir+aNameOutTmp;
        //cout<<strCpImg<<endl;
        System(strCpImg);

        RotateImgBy90Deg(aDir, aNameOutTmp, aNameOut);
        strCpImg = "rm "+aDir+aNameOutTmp;
        //cout<<strCpImg<<endl;
        System(strCpImg);
    }
}

int BruteForce(int argc,char ** argv, const std::string &aArg="")
{
    cCommonAppliTiepHistorical aCAS3D;

    std::string aImg1;
    std::string aImg2;

    std::string aType;

    bool bRotate = false;

    ElInitArgMain
     (
         argc,argv,
         LArgMain()
                << EAMC(aType,"BruteForce or Guided")
                     << EAMC(aImg1,"First image name")
                     << EAMC(aImg2,"Second image name"),
         LArgMain()
                     << EAM(bRotate,"Rotate",true,"Rotate the left image by 90 degree 4 times for methods not invariant to rotation (e.g. SuperGlue), Def=false")
                     << aCAS3D.ArgBasic()
                     << aCAS3D.ArgGetPatchPair()
     );

    aCAS3D.mOutDir += "/";
    ELISE_fp::MkDir(aCAS3D.mOutDir);

    std::string aOutImg1 = GetFolderName(aImg1) + "." + StdPostfix(aImg1);
    std::string aOutImg2 = GetFolderName(aImg2) + "." + StdPostfix(aImg2);

   if(true)
   {
       cElComposHomographie aUnitHX(1, 0, 0);
       cElComposHomographie aUnitHY(0, 1, 0);
       cElComposHomographie aUnitHZ(0, 0, 1);
       cElHomographie  aUnitH =  cElHomographie(aUnitHX,aUnitHY,aUnitHZ);

       //no rotation
       GetTilePair(aCAS3D.mOutDir, aOutImg1, aOutImg1, aOutImg2, aImg1, aImg2, aCAS3D.mPatchSz, aCAS3D.mBufferSz, aCAS3D.mImgPair, aCAS3D.mDir, aCAS3D.mSubPatchXml, aImg1, aUnitH);
   }

   if(bRotate == true)
   {
       Tiff_Im aDSMIm1((aCAS3D.mDir+aImg1).c_str());
       Pt2di ImgSzL = aDSMIm1.sz();

       std::string aImg1_Rotate;
       std::string aOutImg1_Rotate;
       std::string aSubPatchXml;
       std::string aImgPair;
       std::string aImgBase = aImg1;

       //rotate 90 degree
       if(true)
       {
           cElComposHomographie aRotateHX(0, 1, 0);
           cElComposHomographie aRotateHY(-1, 0, ImgSzL.y);
           cElComposHomographie aRotateHZ(0, 0, 1);
           cElHomographie  aRotateH =  cElHomographie(aRotateHX,aRotateHY,aRotateHZ);
           aRotateH.Show();

           aImg1_Rotate = StdPrefix(aImgBase)+"_R90."+StdPostfix(aImgBase);
           aOutImg1_Rotate = StdPrefix(aOutImg1)+"_R90."+StdPostfix(aOutImg1);
           aSubPatchXml = StdPrefix(aCAS3D.mSubPatchXml)+"_R90."+StdPostfix(aCAS3D.mSubPatchXml);
           aImgPair = StdPrefix(aCAS3D.mImgPair)+"_R90."+StdPostfix(aCAS3D.mImgPair);

           //cout<<aImg1_Rotate<<",,,"<<aSubPatchXml<<",,,"<<aImgPair<<endl;

           RotateImgBy90Deg(aCAS3D.mDir, aImgBase, aImg1_Rotate);
           GetTilePair(aCAS3D.mOutDir, aOutImg1, aOutImg1_Rotate, aOutImg2, aImg1_Rotate, aImg2, aCAS3D.mPatchSz, aCAS3D.mBufferSz, aImgPair, aCAS3D.mDir, aSubPatchXml, aImg1, aRotateH);
       }

       //rotate 180 degree
       if(true)
       {
           cElComposHomographie aRotateHX(-1, 0, ImgSzL.x);
           cElComposHomographie aRotateHY(0, -1, ImgSzL.y);
           cElComposHomographie aRotateHZ(0, 0, 1);
           cElHomographie  aRotateH =  cElHomographie(aRotateHX,aRotateHY,aRotateHZ);
           aRotateH.Show();

           aImg1_Rotate = StdPrefix(aImgBase)+"_R180."+StdPostfix(aImgBase);
           aOutImg1_Rotate = StdPrefix(aOutImg1)+"_R180."+StdPostfix(aOutImg1);
           aSubPatchXml = StdPrefix(aCAS3D.mSubPatchXml)+"_R180."+StdPostfix(aCAS3D.mSubPatchXml);
           aImgPair = StdPrefix(aCAS3D.mImgPair)+"_R180."+StdPostfix(aCAS3D.mImgPair);

           //cout<<aImg1_Rotate<<",,,"<<aSubPatchXml<<",,,"<<aImgPair<<endl;

           RotateImgBy90DegNTimes(aCAS3D.mDir, aImgBase, aImg1_Rotate, 2);
           GetTilePair(aCAS3D.mOutDir, aOutImg1, aOutImg1_Rotate, aOutImg2, aImg1_Rotate, aImg2, aCAS3D.mPatchSz, aCAS3D.mBufferSz, aImgPair, aCAS3D.mDir, aSubPatchXml, aImg1, aRotateH);
       }

       //rotate 270 degree
       if(true)
       {
           cElComposHomographie aRotateHX(0, -1, ImgSzL.x);
           cElComposHomographie aRotateHY(1, 0, 0);
           cElComposHomographie aRotateHZ(0, 0, 1);
           cElHomographie  aRotateH =  cElHomographie(aRotateHX,aRotateHY,aRotateHZ);
           aRotateH.Show();

           aImg1_Rotate = StdPrefix(aImgBase)+"_R270."+StdPostfix(aImgBase);
           aOutImg1_Rotate = StdPrefix(aOutImg1)+"_R270."+StdPostfix(aOutImg1);
           aSubPatchXml = StdPrefix(aCAS3D.mSubPatchXml)+"_R270."+StdPostfix(aCAS3D.mSubPatchXml);
           aImgPair = StdPrefix(aCAS3D.mImgPair)+"_R270."+StdPostfix(aCAS3D.mImgPair);

           //cout<<aImg1_Rotate<<",,,"<<aSubPatchXml<<",,,"<<aImgPair<<endl;

           RotateImgBy90DegNTimes(aCAS3D.mDir, aImgBase, aImg1_Rotate, 3);
           GetTilePair(aCAS3D.mOutDir, aOutImg1, aOutImg1_Rotate, aOutImg2, aImg1_Rotate, aImg2, aCAS3D.mPatchSz, aCAS3D.mBufferSz, aImgPair, aCAS3D.mDir, aSubPatchXml, aImg1, aRotateH);
       }
    }

    return 0;
}

int Guided(int argc,char ** argv, const std::string &aArg="")
{
    cCommonAppliTiepHistorical aCAS3D;

    std::string aImg1;
    std::string aImg2;
    std::string aOri1;
    std::string aOri2;

    std::string aType;

    std::string aPara3DH = "";

    bool bPrint = false;

    std::string aDSMDirL = "";
    std::string aDSMFileL;
    aDSMFileL = "MMLastNuage.xml";
    std::string aPrefix = "";

    ElInitArgMain
     (
         argc,argv,
         LArgMain()
                << EAMC(aType,"BruteForce or Guided")
                     << EAMC(aImg1,"First image name")
                     << EAMC(aImg2,"Second image name")
                     << EAMC(aOri1,"Orientation of first image")
                     << EAMC(aOri2,"Orientation of second image"),
         LArgMain()
                     << aCAS3D.ArgBasic()
                     << aCAS3D.ArgGetPatchPair()
                << EAM(aPara3DH, "Para3DH", false, "Input xml file that recorded the paremeter of the 3D Helmert transformation from orientation of first image to second image, Def=none")
                << EAM(bPrint, "Print", false, "Print corner coordinate, Def=false")
                << EAM(aDSMDirL, "DSMDirL", true, "DSM directory of first image, Def=none")
                << EAM(aDSMFileL, "DSMFileL", true, "DSM File of first image, Def=MMLastNuage.xml")
                << EAM(aPrefix, "Prefix", true, "The prefix for the name of images (for debug only), Def=none")

     );

    aCAS3D.mOutDir += "/";
    ELISE_fp::MkDir(aCAS3D.mOutDir);

    StdCorrecNameOrient(aOri1,"./",true);
    StdCorrecNameOrient(aOri2,"./",true);

     std::string aKeyOri1 = "NKS-Assoc-Im2Orient@-" + aOri1;
     std::string aKeyOri2 = "NKS-Assoc-Im2Orient@-" + aOri2;

     std::string aIm1OriFile = aCAS3D.mICNM->Assoc1To1(aKeyOri1,aImg1,true);
     std::string aIm2OriFile = aCAS3D.mICNM->Assoc1To1(aKeyOri2,aImg2,true);


    cTransform3DHelmert aTrans3DH(aPara3DH);

    std::string aOutImg1 = GetFileName(aImg1);
    std::string aOutImg2 = GetFileName(aImg2);
    GetPatchPair(aCAS3D.mOutDir, aOutImg1, aOutImg2, aImg1, aImg2, aIm1OriFile, aIm2OriFile, aCAS3D.mPatchSz, aCAS3D.mBufferSz, aPrefix + aCAS3D.mImgPair, aCAS3D.mDir, aPrefix + aCAS3D.mSubPatchXml, aTrans3DH, aDSMFileL, aDSMDirL, bPrint, aPrefix);

    return 0;
}

int GetPatchPair_main(int argc,char ** argv)
{
    bool aModeHelp=true;
    eGetPatchPair_HistoP aType=eNbTypePPHP;
    StdReadEnum(aModeHelp,aType,argv[1],eNbTypePPHP);

   std::string TheType = argv[1];

   if (TheType == "BruteForce")
   {
       int aRes = BruteForce(argc, argv, TheType);
       return aRes;
   }
   else if (TheType == "Guided")
   {
       int aRes = Guided(argc, argv, TheType);
       return aRes;
   }

   return EXIT_SUCCESS;
}
