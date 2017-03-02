#include "EsSimilitude.h"

struct Helm2DS{
 double Rotx;
 double Roty;
 double Tx;
 double Ty;
 double Scale;
};

int ProcessThmImgs_main(int argc,char ** argv)
{
    std::string aFullName, aXml, aDir, aPat, aImMasq, aNameMasq, aOut="All_H2D.txt";

    int aSzW=10;
    double aRegul=4.0;
    bool useDequant=false;
    double aIncCalc=4.0;
    int aSsResolOpt=4;
    Pt2dr aPxMoy(0,0);
    int  aZoomInit=4;
    bool aPurge=true;
    int aPas=1;
    int aFonc=0;

    ElInitArgMain
    (
         argc, argv,
         LArgMain() << EAMC(aFullName,"Full Name (Dir+Pat)")
                    << EAMC(aXml,"Input .xml file for MICMAC"),
         LArgMain() << EAM(aImMasq,"Masq",true,"Mask of focus zone (def=none)", eSAM_IsExistFile)
                    << EAM(aSzW,"SzW",true,"Size of window (Def=10, mean 30x30)")
                    << EAM(aRegul,"Reg",true,"Regularization (Def=4.0)")
                    << EAM(useDequant,"Dequant",true,"Dequantify (Def=true)")
                    << EAM(aIncCalc,"Inc",true,"Initial uncertainty (Def=4.0)")
                    << EAM(aSsResolOpt,"SsResolOpt",true,"Merging factor (Def=4)")
                    << EAM(aPxMoy,"PxMoy",true,"Px-Moy , Def=(0,0)")
                    << EAM(aZoomInit,"ZoomInit",true,"Initial Zoom, Def=4 (can be long of Inc>2)")
                    << EAM(aOut,"Out",false,"Output File Name for All Helmert2D Params ; Def=All_H2D.txt")
                    << EAM(aPurge,"Purge",true,"Purge all .txt file from EsSim")
                    << EAM(aPas,"Pas",true,"interval of images for correlation ; Def=1")
                    << EAM(aFonc,"Fonc",true,"Choice of fonctions wanted to be exacuted; Def=0(correlation+similitude estimation), 1(correlation), 2(similitude estimation)")
    );

    SplitDirAndFile(aDir, aPat, aFullName);
    SplitDirAndFile(aDir, aNameMasq, aImMasq);

    cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);
    std::vector<std::string> aLFile = *(aICNM->Get(aPat));

    std::vector<std::string> aVComMM, aVComESS, aVOFESS;

    if (IsPostfixed(aNameMasq)) aNameMasq = StdPrefixGen(aNameMasq);

    for( unsigned int aK=aPas; aK<=aLFile.size()-aPas; aK=aK+aPas)
    {

        std::string aDirMEC = "MEC-" + aLFile.at(0) + "-" + aLFile.at(aK);
        MakeFileDirCompl(aDirMEC);

        std::string aCom1 =    MM3dBinFile("MICMAC")
                                    + aXml
                                    + " WorkDir=" + aDir
                                    //+ " +DirMEC=" + aDirMEC
                                    + " +Im1=" + aLFile.at(0)
                                    + " +Im2=" + aLFile.at(aK)
                                    //+ " +Masq=" + aNameMasq
                                    + " +SzW=" + ToString(aSzW)
                                    + " +RegulBase=" + ToString(aRegul)
                                    + " +Inc=" + ToString(aIncCalc)
                                    + " +SsResolOpt=" + ToString(aSsResolOpt)
                                    + " +Px1Moy=" + ToString(aPxMoy.x)
                                    + " +Px2Moy=" + ToString(aPxMoy.y)
                                    + " +ZoomInit=" + ToString(aZoomInit)
                                    ;



        if (EAMIsInit(&aImMasq))
        {
            ELISE_ASSERT(aDir==DirOfFile(aImMasq),"Image not on same directory !!!");
            MakeMetaData_XML_GeoI(aImMasq);
            std::cout << "aImMasq =" << aImMasq << std::endl;
            aCom1 = aCom1 + " +UseMasq=true +Masq=" + StdPrefix(aImMasq);
        }

        if (useDequant)
        {
            aCom1 = aCom1 + " +UseDequant=true";
        }

        std::string aOutEss = aLFile.at(0) + "-" + aLFile.at(aK) +".txt";
        std::string aCom2 = MM3dBinFile("TestLib EsSim ")
                             + aDirMEC
                             + "Px1_Num7_DeZoom1_LeChantier.tif "
                             + aDirMEC
                             + "Px2_Num7_DeZoom1_LeChantier.tif"
                             + " aNbGrill=[50,50] ExportTxt=1 "
                             + "Out="
                             + aOutEss;


        aVComMM.push_back(aCom1); //vector of commands of MICMAC
        aVComESS.push_back(aCom2); //vector of commands of EsSim
        aVOFESS.push_back(aOutEss); //vector of names of output files from EsSim
    }


    if (!MMVisualMode)
    {
		//LG : Fix compile (aIm1 and aIm2 are not defined anymore, replaced by aLFile.at(0) and aLFile.at(1), possibly right????????
        #if (ELISE_windows)
			//replace( aIm1.begin(), aIm1.end(), '\\', '/');
			//replace( aIm2.begin(), aIm2.end(), '\\', '/');
			replace(aLFile.at(0).begin(), aLFile.at(0).end(), '\\', '/');
			replace(aLFile.at(1).begin(), aLFile.at(1).end(), '\\', '/');
            replace( aImMasq.begin(), aImMasq.end(), '\\', '/' );
        #endif

       for(unsigned int aP=0; aP<aVComMM.size(); aP++)
       {

           std::cout << "***************************************" << std::endl;
           if(aFonc==0 || aFonc==1)
           {
                std::cout << "aComMM = " << aVComMM.at(aP) << std::endl;
           }
           if(aFonc==0 || aFonc==2)
           {
                std::cout << "aComESS = " << aVComESS.at(aP) << std::endl;
           }

           std::cout << "***************************************" << std::endl;

           if(aFonc==0 || aFonc==1)
           {
                system_call(aVComMM.at(aP).c_str());
           }
           if(aFonc==0 || aFonc==2)
           {
                system_call(aVComESS.at(aP).c_str());
           }
        }
    }

    std::vector<Helm2DS> aVHC;
    for(unsigned int aF=0; aF<aVOFESS.size(); aF++)
    {
        //read all files
        ifstream aFichier(aVOFESS.at(aF).c_str());

        if(aFichier)
        {
            std::string aLigne;

            while(!aFichier.eof())
            {
                getline(aFichier,aLigne);

                if(!aLigne.empty() && aLigne.compare(0,1,"#") != 0)
                {
                    char *aBuffer = strdup((char*)aLigne.c_str());
                    std::string aImg1 = strtok(aBuffer," ");
                    std::string aImg2 = strtok( NULL, " " );
                    char *aA = strtok( NULL, " " );
                    char *aB = strtok( NULL, " " );
                    char *aC = strtok( NULL, " " );
                    char *aD = strtok( NULL, " " );
                    char *aS = strtok( NULL, " " );

                    //keep only second line information
                    Helm2DS aHC;
                    aHC.Rotx = atof(aA);
                    aHC.Roty = atof(aB);
                    aHC.Tx = atof(aC);
                    aHC.Ty = atof(aD);
                    aHC.Scale = atof(aS);

                    aVHC.push_back(aHC);
                }

            }

            aFichier.close();
        }

        else
        {
            std::cout<< "Error While opening file" << '\n';
        }

    }

    //put them all in one unique file
    if (!MMVisualMode)
    {
            FILE * aFP = FopenNN(aOut,"w","ProcessThmImgs_main");
            cElemAppliSetFile aEASF(aDir + ELISE_CAR_DIR + aOut);

            std::string Format = "#F=R1 R2 Tx Ty Scale";
            fprintf(aFP,"%s\n", Format.c_str());

            for(unsigned int aH=0; aH<aVHC.size(); aH++)
            {
                fprintf(aFP,"%f %f %f %f %f\n", aVHC.at(aH).Rotx, aVHC.at(aH).Roty,  aVHC.at(aH).Tx,  aVHC.at(aH).Ty,  aVHC.at(aH).Scale);
            }

            ElFclose(aFP);
    }

    //purge all .txt files from EsSim
    for(unsigned int aF=0; aF<aVOFESS.size(); aF++)
    {
        ELISE_fp::RmFile(aVOFESS.at(aF));
    }


    return EXIT_SUCCESS;
}

int EsSim_main(int argc,char ** argv)
{
    string aImgX, aImgY, aDir, aPat1, aPat2, aOut="";
    Pt3di aSzDisp(50,50,5);
    Pt2dr aPtCtr(0,0);
    int aSzW = 5;
    int nInt = 0;
    Pt2di aNbGrill(1,1);
    double aSclDepl = 50.0;
    bool aExportTxt=false;
    bool aSaveImg=false;

    ElInitArgMain
            (
                argc,argv,
                //mandatory arguments
                LArgMain()
                //<< EAMC(aDir, "Dir", eSAM_None)
                << EAMC(aImgX, "Img de deplacement X", eSAM_IsExistFile)
                << EAMC(aImgY, "Img de deplacement Y", eSAM_IsExistFile),

                //optional arguments
                LArgMain()
                << EAM(aSzDisp, "SzDisp", true, "Size Display Win ; Def=[50,50,5]")
                << EAM(aPtCtr, "Pt", true, "Pt Correl central ; Def=[0,0]")
                << EAM(aSzW, "aSzW", true, "Sz win Correl (demi) ; Def=5")
                << EAM(aSaveImg, "SaveImg", true,"Save Image when Disp")
                << EAM(aNbGrill, "aNbGrill", true, "Nb de Grill ; Def=[1,1]")
                << EAM(aSclDepl, "Scl", true, "Scale Factor Vec Deplacement ; View Only ; Def=50")
                << EAM(aExportTxt, "ExportTxt", true, "Export results of Helmert2D in .txt format ; Def=false")
                << EAM(aOut, "Out", true, "Output file name for A,B,C,D Helmert2D Params ; Def=Img1_Img2.txt")
             );

    SplitDirAndFile(aDir, aPat1, aImgX);
    SplitDirAndFile(aDir, aPat2, aImgY);

    if(aOut == "")
    {
        aOut = StdPrefix(aPat1) + "_" + StdPrefix(aPat2) + ".txt";
    }

    if (MMVisualMode)     return EXIT_SUCCESS;

    if (EAMIsInit(&aSzDisp))
    {
        nInt = 1;
    }

    cParamEsSim * aParam = new cParamEsSim(aDir, aPat1, aPat2, aPtCtr, aSzW, aSzDisp, nInt, aNbGrill, aSclDepl, aSaveImg);
    cAppliEsSim * aAppli = new cAppliEsSim(aParam);

    Pt2dr aRotCosSin;
    Pt2dr aTransXY;

    if ( !EAMIsInit(&aNbGrill) && nInt == 0)
    {
        ElPackHomologue aPack;
        aAppli->getHomolInVgt(aPack, aPtCtr, aSzW);
        aAppli->EsSimFromHomolPack(aPack, aRotCosSin, aTransXY);
    }

    if (EAMIsInit(&aNbGrill))
    {
        aAppli->EsSimEnGrill(aAppli->VaP0Grill(), aSzW, aRotCosSin, aTransXY);
    }

    if ( !EAMIsInit(&aNbGrill) && nInt == 1)
    {
        aAppli->EsSimAndDisp(aPtCtr, aSzW, aRotCosSin, aTransXY);
    }

    if (!MMVisualMode)
    {
        if(aExportTxt)
        {
            FILE * aFP = FopenNN(aOut,"w","EsSim_main");
            cElemAppliSetFile aEASF(aDir + ELISE_CAR_DIR + aOut);

            std::string Format = "#F=IMG1 IMG2 R1 R2 Tx Ty Scale";
            fprintf(aFP,"%s\n", Format.c_str());

            double aScale = aRotCosSin.x > 0 ? euclid(aRotCosSin) : -euclid(aRotCosSin);

            fprintf(aFP,"%s %s %f %f %f %f %f\n", aPat1.c_str(),aPat2.c_str(), aRotCosSin.x/aScale, aRotCosSin.y/aScale, aTransXY.x, aTransXY.y, aScale);

            ElFclose(aFP);

        }
    }

 return EXIT_SUCCESS;
}
