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
#include "cnpy.h"
#include "cnpy.cpp"



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

bool CheckFile(std::string input_dir, std::string SH, std::string input_pairs)
{
    ifstream in(input_dir+input_pairs);
    std::string s;

    while(getline(in,s))
    {
        std::string aImg1,aImg2;
        std::stringstream is(s);
        is>>aImg1>>aImg2;

        std::string aSHDir = input_dir + "/Homol" + SH+"/";
        ELISE_fp::MkDir(aSHDir);
        std::string aNewDir = aSHDir + "Pastis" + aImg1;
        ELISE_fp::MkDir(aNewDir);
        std::string aNameFile1 = aNewDir + "/"+aImg2+".txt";

        aNewDir = aSHDir + "Pastis" + aImg2;
        ELISE_fp::MkDir(aNewDir);
        std::string aNameFile2 = aNewDir + "/"+aImg1+".txt";

        if (ELISE_fp::exist_file(aNameFile1) == false || ELISE_fp::exist_file(aNameFile2) == false)
        {
            return false;
        }
    }
    return true;
}

void Npz2Homol(Pt2di resize, std::string input_dir, std::string SH, std::string input_pairs, bool keepNpzFile)
{
    ifstream in(input_dir+input_pairs);
    std::string s;
    while(getline(in,s))
    {
        std::string aImg1,aImg2;
        std::stringstream is(s);
        is>>aImg1>>aImg2;

        //load the entire npz file
        std::string aFileName = input_dir+aImg1.substr(0, aImg1.rfind("."))+"_"+aImg2.substr(0, aImg2.rfind("."))+"_matches.npz";
        //cout<<aFileName<<endl;

        if (ELISE_fp::exist_file(aFileName) == false)
        {
            cout<<aFileName<<" didn't exist, hence skipped.\n";
            continue;
        }

        Pt2dr ptScaleL = Pt2dr(1.0, 1.0);
        Pt2dr ptScaleR = Pt2dr(1.0, 1.0);

        Tiff_Im aRGBIm1((input_dir+aImg1).c_str());
        Pt2di ImgSzL = aRGBIm1.sz();
        Tiff_Im aRGBIm2((input_dir+aImg2).c_str());
        Pt2di ImgSzR = aRGBIm2.sz();

        //cout<<"Left img size: "<<ImgSzL.x<<", "<<ImgSzL.y<<endl;
        //cout<<"Right img size: "<<ImgSzR.x<<", "<<ImgSzR.y<<endl;

        //scale the pt if the images are resized before applying SuperGlue
        if(resize.x > 0 && resize.y > 0)
        {
            ptScaleL.x = 1.0*ImgSzL.x/resize.x;
            ptScaleL.y = 1.0*ImgSzL.y/resize.y;
            ptScaleR.x = 1.0*ImgSzR.x/resize.x;
            ptScaleR.y = 1.0*ImgSzR.y/resize.y;
        }

        cnpy::npz_t my_npz = cnpy::npz_load(aFileName);

        cnpy::NpyArray arr_keypt0_raw = my_npz["keypoints0"];
        float* arr_keypt0 = arr_keypt0_raw.data<float>();

        cnpy::NpyArray arr_keypt1_raw = my_npz["keypoints1"];
        float* arr_keypt1 = arr_keypt1_raw.data<float>();
        int nPtNum1 = arr_keypt1_raw.shape[0];

        if(arr_keypt0_raw.shape[0]<=0 || arr_keypt1_raw.shape[0]<=0)
        {
            cout<<aFileName<<" has no key point in left or right image, hence skipped.\n";
            continue;
        }

        cnpy::NpyArray arr_matches_raw = my_npz["matches"];
        long * arr_matches = arr_matches_raw.data<long>();
        int nMatchNum = arr_matches_raw.shape[0];
        //cout<<"nMatchNum: "<<nMatchNum<<endl;
        int i;
        ElPackHomologue aPack;
        int nValidMatchNum = 0;
        for(i=0; i<nMatchNum; i++)
        {
            long nMatchVal = arr_matches[i];
/*
            if(nMatchVal != -1)
                cout<<i<<": "<<nMatchVal<<endl;
            //continue;
*/
            //the sizes of arr_keypt0, arr_keypt1 and arr_matches are: Num1*2, Num2*2, Num1*1
            //-1 means no match for arr_keypt0[i]
            if(nMatchVal != -1)
            {
                if(nMatchVal < 0 || nMatchVal >= nPtNum1)
                    continue;
                Pt2dr ptL = Pt2dr(arr_keypt0[i*2]*ptScaleL.x, arr_keypt0[i*2+1]*ptScaleL.y);
                Pt2dr ptR = Pt2dr(arr_keypt1[nMatchVal*2]*ptScaleR.x, arr_keypt1[nMatchVal*2+1]*ptScaleR.y);
                aPack.Cple_Add(ElCplePtsHomologues(ptL, ptR));
                nValidMatchNum++;
            }
        }
        //break;

        //cout<<"nValidMatchNum: "<<nValidMatchNum<<endl;

        std::string aCom = "mm3d SEL" + BLANK + input_dir + BLANK + aImg1 + BLANK + aImg2 + BLANK + "KH=NT SzW=[600,600] SH="+SH;
        cout<<aCom<<endl;
        cout<<"tie point number: "<<nValidMatchNum<<endl;

        std::string aSHDir = input_dir + "/Homol" + SH+"/";
        ELISE_fp::MkDir(aSHDir);
        std::string aNewDir = aSHDir + "Pastis" + aImg1;
        ELISE_fp::MkDir(aNewDir);
        std::string aNameFile1 = aNewDir + "/"+aImg2+".txt";

        aNewDir = aSHDir + "Pastis" + aImg2;
        ELISE_fp::MkDir(aNewDir);
        std::string aNameFile2 = aNewDir + "/"+aImg1+".txt";

        FILE * fpTiePt1 = fopen(aNameFile1.c_str(), "w");
        FILE * fpTiePt2 = fopen(aNameFile2.c_str(), "w");
        for (ElPackHomologue::iterator itCpl=aPack.begin();itCpl!=aPack.end() ; itCpl++)
        {
            ElCplePtsHomologues tiept = itCpl->ToCple();
            fprintf(fpTiePt1, "%lf %lf %lf %lf\n", tiept.P1().x, tiept.P1().y, tiept.P2().x, tiept.P2().y);
            fprintf(fpTiePt2, "%lf %lf %lf %lf\n", tiept.P2().x, tiept.P2().y, tiept.P1().x, tiept.P1().y);
        }
        fclose(fpTiePt1);
        fclose(fpTiePt2);

        if(keepNpzFile == false)
        {
            std::string cmmd = "rm " + aFileName;
            System(cmmd);
        }
    }
}

void ReadImgPairs(std::string input_dir, std::string input_pairs)
{
    ifstream in(input_dir+input_pairs);
    std::string s;
    while(getline(in,s))
    {
        //printf("%s\n", s.c_str());
        std::string aImg1,aImg2;
        std::stringstream is(s);
        is>>aImg1>>aImg2;
        //cout<<str2<<","<<str3<<endl;

    }
}

int SuperGlue_main(int argc,char ** argv)
{
   cCommonAppliTiepHistorical aCAS3D;

   std::string input_pairs;

   bool bCheckFile = false;

   ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(input_pairs, "txt file that listed the image pairs"),
        LArgMain()
                    //<< aCAS3D.ArgBasic()
                    << aCAS3D.ArgSuperGlue()
                    << EAM(bCheckFile, "CheckFile", true, "Check if the result files of inter-epoch correspondences exist (if so, skip to avoid repetition), Def=false")

    );

   std::string strOpt = aCAS3D.mStrOpt;
   std::string strMicMacDirBin = aCAS3D.mStrEntSpG;
   std::string cmmd;
   if(strMicMacDirBin.length()==0)
   {
       //strMicMacDir = MMBinFile(MM3DStr);
       //strMicMacDirBin = strMicMacDirBin.substr(0, strMicMacDir.length()-9) + "src/uti_phgrm/TiePHistorical/SuperGluePretrainedNetwork-master/match_pairs.py";
       strMicMacDirBin = MMBinFile(MM3DStr);
       std::string strMicMacDirTPHisto = strMicMacDirBin.substr(0, strMicMacDirBin.length()-9) + "src/uti_phgrm/TiePHistorical/";
       cmmd = "bash " + strMicMacDirTPHisto + "run.sh --input_pairs "+aCAS3D.mInput_dir+input_pairs+" --input_dir "+aCAS3D.mInput_dir+" --output_dir "+aCAS3D.mOutput_dir + " --max_keypoints "+std::to_string(aCAS3D.mMax_keypoints);

   }
   else
   {
       cmmd = strMicMacDirBin + " --input_pairs "+aCAS3D.mInput_dir+input_pairs+" --input_dir "+aCAS3D.mInput_dir+" --output_dir "+aCAS3D.mOutput_dir + " --max_keypoints "+std::to_string(aCAS3D.mMax_keypoints);
   }

   bool bExe = true;
   if(bCheckFile == true)
   {
       bool bFileExist = CheckFile(aCAS3D.mInput_dir, aCAS3D.mSpGlueOutSH, input_pairs);
       if(bFileExist == true)
       {
           printf("%s: Result files already exist, hence skipped\n", input_pairs.c_str());
           bExe = false;
       }
   }


   if(bExe)
   {
        //std::string cmmd = strMicMacDir + " --input_pairs "+aCAS3D.mInput_dir+input_pairs+" --input_dir "+aCAS3D.mInput_dir+" --output_dir "+aCAS3D.mOutput_dir + " --max_keypoints "+std::to_string(aCAS3D.mMax_keypoints);
       //std::string cmmd = "/home/lulin/Documents/ThirdParty/SuperGluePretrainedNetwork-master/match_pairs.py --input_pairs "+aCAS3D.mInput_dir+input_pairs+" --input_dir "+aCAS3D.mInput_dir+" --output_dir "+aCAS3D.mOutput_dir + " --max_keypoints "+std::to_string(aCAS3D.mMax_keypoints);
       if(aCAS3D.mViz == true)
           cmmd += " --viz";
       if(aCAS3D.mModel == "indoor")
           cmmd += " --superglue indoor";
       else
           cmmd += " --superglue outdoor";
       if(aCAS3D.mResize.x > 0 && aCAS3D.mResize.y > 0)
           cmmd += " --resize " + std::to_string(aCAS3D.mResize.x) + " " + std::to_string(aCAS3D.mResize.y);
       else
           cmmd += " --resize -1";
       cmmd += " " + strOpt;

       printf("%s\n", cmmd.c_str());
       System(cmmd);


       //todefine(aCAS3D.input_dir, aCAS3D.output_dir, input_pairs);
       //ReadImgPairs(aCAS3D.input_dir, input_pairs);
       Npz2Homol(aCAS3D.mResize, aCAS3D.mInput_dir, aCAS3D.mSpGlueOutSH, input_pairs, aCAS3D.mKeepNpzFile);
    }
   return EXIT_SUCCESS;
}
