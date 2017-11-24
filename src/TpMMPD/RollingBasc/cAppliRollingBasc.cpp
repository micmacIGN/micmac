#include "RollingBasc.h"
#include <chrono>
#include <ctime>

int FlagOfDeg(const std::vector<std::string> & aDXY);
int FlagOfDeg(const Pt3di & aDXY);


int GCPRollingBasc_main(int argc,char ** argv)
{
    std::string  aDir,aPat,aFullDir;
    std::string AeroIn;
    std::string DicoPts;
    std::string MesureIm;
    bool        ModeL1 = false;
    bool        CPI = false;
    bool ShowUnused = true;
    bool ShowDetail = false;
    bool NLDShow = false;
    bool NLDFTR = true;
    std::string ForceSol;

    std::string aTXTRollCtrlSuffix {"RollCtrl.txt"};

    std::string aPatNLD;
    std::vector<std::string> NLDDegX;  NLDDegX.push_back("1");  NLDDegX.push_back("X"); NLDDegX.push_back("Y");
    std::vector<std::string> NLDDegY;  NLDDegY.push_back("1");  NLDDegY.push_back("X"); NLDDegY.push_back("Y");
    std::vector<std::string> NLDDegZ;  NLDDegZ.push_back("1");  NLDDegZ.push_back("X"); NLDDegZ.push_back("X2");

    std::string OutRollResTxt {""};
    ElInitArgMain
    (
        argc,argv,
        LArgMain()  << EAMC(aFullDir,"Full name (Dir+Pat)", eSAM_IsPatFile)
                    << EAMC(AeroIn,"Orientation in", eSAM_IsExistDirOri)
                    << EAMC(DicoPts,"Ground Control Points File", eSAM_IsExistFile)
                    << EAMC(MesureIm,"Image Measurements File", eSAM_IsExistFile),
        LArgMain()
                    <<  EAM(ModeL1,"L1",true,"L1 minimisation vs L2 (Def=false)", eSAM_IsBool)
                    <<  EAM(CPI,"CPI",true,"when Calib Per Image has to be used", eSAM_IsBool)
                    <<  EAM(ShowUnused,"ShowU",true,"Show unused point (def=true)", eSAM_IsBool)
                    <<  EAM(ShowDetail,"ShowD",true,"Show details (def=false)", eSAM_IsBool)
                    <<  EAM(aPatNLD,"PatNLD",true,"Pattern for Non linear deformation, with aerial like geometry (def,unused)")
                    <<  EAM(NLDDegX,"NLDegX",true,"Non Linear monoms for X, when PatNLD, (Def =[1,X,Y])")
                    <<  EAM(NLDDegY,"NLDegY",true,"Non Linear monoms for Y, when PatNLD, (Def =[1,X,Y])")
                    <<  EAM(NLDDegZ,"NLDegZ",true,"Non Linear monoms for Z, when PatNLD, (Def =[1,X,X2])")
                    <<  EAM(NLDFTR,"NLFR",true,"Non Linear : Force True Rot (Def=true)",eSAM_IsBool)
                    <<  EAM(NLDShow,"NLShow",true,"Non Linear : Show Details (Def=false)",eSAM_IsBool)
                    <<  EAM(ForceSol,"ForceSol",true,"To Force Sol from existing solution (xml file)",eSAM_IsExistFile)
                    <<  EAM(OutRollResTxt,"OutTxt",true,"Rolling Bascule And Control Result Txt Output")
    );

    auto timerStart = std::chrono::system_clock::now();

    SplitDirAndFile(aDir,aPat,aFullDir);
    StdCorrecNameOrient(AeroIn,aDir);


    MMD_InitArgcArgv(argc,argv);

    std::size_t extensionXML = DicoPts.find_last_of(".");

    std::cout<<"Name Dico : "<<DicoPts.substr(0, extensionXML)<<std::endl;

    string FileDicoPts{DicoPts.substr(0, extensionXML)}; // separate DicoPts file name with extension

    std::string TmpDicoPtsXML {"Tmp-RollBasc_" + FileDicoPts +"/"};

    if (ELISE_fp::IsDirectory(TmpDicoPtsXML) )
    {
        ELISE_fp::PurgeDirRecursif(TmpDicoPtsXML);  //rmDir ne marche pas
    }

    ELISE_fp::MkDirSvp(TmpDicoPtsXML);

    /* ===== Make GCP Rolling Bascule & Ctrl XML File ===== */
    cDicoAppuisFlottant aDico = StdGetFromPCP(DicoPts, DicoAppuisFlottant);
    cout<<"Dico Imported : NbPts "<<aDico.OneAppuisDAF().size()<<endl;

    list<cOneAppuisDAF>::iterator it1, it2;

    it1 = aDico.OneAppuisDAF().begin();

    std::vector<string> VNameGCPRollFile;

    std::list<std::string> VComBasc;

    std::list<std::string> VComCtrl;

    std::vector<std::string> VDelFol;

    std::string OutTxt {TmpDicoPtsXML + "Ctrl_" + FileDicoPts + /*"_" + (*it1).NamePt() + */".txt"};

    std::string OutTxtOnlyName {"Ctrl_" + FileDicoPts + /*"_" + (*it1).NamePt() + */".txt"};

    while (it1 != aDico.OneAppuisDAF().end())
    {
        cDicoAppuisFlottant  aDicoRoll;

        cDicoAppuisFlottant  aDicoCtrl;

        string GCPBascFile {TmpDicoPtsXML + FileDicoPts + "_" + (*it1).NamePt() + ".xml"};

        string GCPCtrlFile {TmpDicoPtsXML + FileDicoPts + "_" + (*it1).NamePt() + "_Ctrl.xml"};

        VNameGCPRollFile.push_back(GCPBascFile);

        for (list<cOneAppuisDAF>::iterator itGCPRoll = aDico.OneAppuisDAF().begin(), end = it1; itGCPRoll != end; ++itGCPRoll)
        {
            cOneAppuisDAF aGCPRoll = *itGCPRoll;
            aDicoRoll.OneAppuisDAF().push_back(aGCPRoll);
        }

        it2 = it1;
        ++it2;

        for (list<cOneAppuisDAF>::iterator itGCPRoll = it2, end = aDico.OneAppuisDAF().end(); itGCPRoll != end; ++itGCPRoll)
        {
            cOneAppuisDAF aGCPRoll = *itGCPRoll;
            aDicoRoll.OneAppuisDAF().push_back(aGCPRoll);
        }


        // ========== Make Bascule Command ============
        std::string AeroOut {"RollBasc_" + FileDicoPts + "_" + (*it1).NamePt()};

        std::string aComBasc =   MM3dBinFile_quotes( "Apero" )
                + ToStrBlkCorr( MMDir()+"include/XML_MicMac/Apero-GCP-Bascule.xml" )+" "
                + std::string(" DirectoryChantier=") +aDir +  std::string(" ")
                + std::string(" +PatternAllIm=") + QUOTE(aPat) + std::string(" ")
                + std::string(" +AeroIn=") + AeroIn
                + std::string(" +AeroOut=") +  AeroOut
                + std::string(" +DicoApp=") +  QUOTE(GCPBascFile)
                + std::string(" +SaisieIm=") +  QUOTE(MesureIm)
                ;

        if (EAMIsInit(&ShowUnused)) aComBasc = aComBasc + " +ShowUnused=" + ToString(ShowUnused);
        if (EAMIsInit(&ShowDetail)) aComBasc = aComBasc + " +ShowDetail=" + ToString(ShowDetail);

        if (EAMIsInit(&ForceSol))
        {
            aComBasc = aComBasc + " +DoForceSol=true  +NameForceSol=" + ForceSol + " ";
        }

        if (ModeL1)
        {
            aComBasc = aComBasc+ std::string(" +L2Basc=") + ToString(!ModeL1);
        }

        if (CPI) aComBasc += " +CPI=true ";


        if (EAMIsInit(&aPatNLD))
        {
            aComBasc = aComBasc + " +UseNLD=true +PatNLD=" + QUOTE(aPatNLD)
                    + " +NLFlagX=" + ToString(FlagOfDeg(NLDDegX))
                    + " +NLFlagY=" + ToString(FlagOfDeg(NLDDegY))
                    + " +NLFlagZ=" + ToString(FlagOfDeg(NLDDegZ))
                    + " +NLDForceTR=" + ToString(NLDFTR)
                    + " +NLDShow=" + ToString(NLDShow)
                    ;
        }

        //std::cout<<endl<<aComBasc<<endl;

        VComBasc.push_back(aComBasc);
        std::string aDelFolder {"Ori-" + AeroOut};
        VDelFol.push_back(aDelFolder);


        // ========== Make Control Command ============

        std::string aComCtrl =   MM3dBinFile_quotes( "Apero" )
                           + ToStrBlkCorr( MMDir()+"include/XML_MicMac/Apero-GCP-Control.xml" )+" "
                           + std::string(" DirectoryChantier=") +aDir +  std::string(" ")
                           + std::string(" +PatternAllIm=") + QUOTE(aPat) + std::string(" ")
                           + std::string(" +AeroIn=") + AeroOut
                           + std::string(" +DicoApp=") +  GCPCtrlFile
                           + std::string(" +SaisieIm=") +  MesureIm
                        ;

        if (EAMIsInit(&ShowUnused)) aComCtrl = aComCtrl + " +ShowUnused=" + ToString(ShowUnused);
        if (CPI) aComCtrl += " +CPI=true ";
        aComCtrl += " +BoolOutTxt=true +OutTxt=";
        aComCtrl += OutTxt;

       //std::cout<<endl<<aComCtrl<<endl;

        VComCtrl.push_back(aComCtrl);
        // ==============================================

        cOneAppuisDAF aGCPRollCtrl = *it1;
        aDicoCtrl.OneAppuisDAF().push_back(aGCPRollCtrl);
        ++it1;

        MakeFileXML(aDicoRoll,GCPBascFile);
        MakeFileXML(aDicoCtrl,GCPCtrlFile);
    }
    // ======= Do Bascule ========
    cEl_GPAO::DoComInParal(VComBasc);
    // ======= Do Control =======
    cEl_GPAO::DoComInSerie(VComCtrl);
    // ======= Do housework =======
    for (uint aKDel {0}; aKDel < VDelFol.size() ; aKDel++)
    {
        if (ELISE_fp::IsDirectory(VDelFol[aKDel]))
        {
            ELISE_fp::RmDir(VDelFol[aKDel]);
        }
    }
    // ======= Do Stat =======
    cout<<"\n"<<"\n"<<"\n";
    cout<<" ======= ROLLING BASCULE CTRL RESULT ======= "<<endl;

    std::string aResRollFile = OutTxt + "_RollCtrl.txt";

    ELISE_ASSERT(ELISE_fp::exist_file(aResRollFile) , "Rolling Bascule Txt Result File not found !")

    std::ifstream infile(aResRollFile);
    std::string line;

    std::string ptDMax, ptDMin;
    std::string ptMaxX, ptMaxY, ptMaxZ;
    double DMax{0.0};
    double DMin{DBL_MAX};
    double Moy{0.0};
    Pt3dr MoyAbs {Pt3dr{0.0,0.0,0.0}};
    Pt3dr Max{Pt3dr{0.0,0.0,0.0}};


    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        string aNamePt;
        double ResX, ResY, ResZ;
        if (!(iss >> aNamePt >> ResX >> ResY >> ResZ)) { break; } // error

        Pt3dr aRes(ResX, ResY, ResZ);

        double aD {euclid(aRes)};

        Moy += aD;
        MoyAbs = aRes.AbsP() + MoyAbs;

        std::cout<<"Ctrl "<<aNamePt<<" , D="<<aD<<" P="<<aRes<<endl;

        if (aD > DMax)
        {
            ptDMax = aNamePt;
            DMax  = aD;
        }
        if (aD < DMin)
        {
            ptDMin = aNamePt;
            DMin = aD;
        }
        if (ResX > Max.x)
        {
            ptMaxX=aNamePt;
            Max.x = ResX;
        }
        if (ResY > Max.y)
        {
            ptMaxY=aNamePt;
            Max.y = ResY;
        }
        if (ResZ > Max.z)
        {
            ptMaxZ=aNamePt;
            Max.z = ResZ;
        }
    }

    infile.close();

    int aNbPt=aDico.OneAppuisDAF().size();
    cout<<" ======= ROLLING BASCULE GCP STAT ======= "<<endl;
    cout<<" === Dist , Moy="<<Moy/double(aNbPt)<<" on NbPt="<<aNbPt<<endl;
    cout<<"            Max="<<DMax<<" on Pt "<<ptDMax<<endl;
    cout<<"            Min="<<DMin<<" on Pt "<<ptDMin<<endl;
    cout<<" === XYZ  , MoyAbs="<<MoyAbs/double(aNbPt)<<" on NbPt="<<aNbPt<<endl;
    cout<<"            Max="<<Max<<" on Pt ["<<ptMaxX<<" , "<<ptMaxY<<" , "<<ptMaxZ<<"]"<<endl;
    cout<<" ========================================"<<endl;


    auto timerEnd = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = timerEnd-timerStart;
    std::time_t end_time = std::chrono::system_clock::to_time_t(timerEnd);




    std::ofstream ofs;
    ofs.open (aResRollFile, std::ofstream::out | std::ofstream::app);

    ofs<<endl;
    ofs<<" ======= ROLLING BASCULE GCP STAT ======= "<<endl;
    ofs<<" === Dist , Moy="<<Moy/double(aNbPt)<<" on NbPt="<<aNbPt<<endl;
    ofs<<"            Max="<<DMax<<" on Pt "<<ptDMax<<endl;
    ofs<<"            Min="<<DMin<<" on Pt "<<ptDMin<<endl;
    ofs<<" === XYZ  , MoyAbs="<<MoyAbs/double(aNbPt)<<" on NbPt="<<aNbPt<<endl;
    ofs<<"            Max="<<Max<<" on Pt ["<<ptMaxX<<" , "<<ptMaxY<<" , "<<ptMaxZ<<"]"<<endl;
    ofs<<" ========================================"<<endl;
    ofs<<endl;
    ofs<<"Rolling Bascule and Control Test on : "<<"\n";
    ofs<<" + Aero : "<<AeroIn<<endl;
    ofs<<" + Pat Im : "<<aFullDir<<endl;
    ofs<<" + Ground Point File : "<<DicoPts<<endl;
    ofs<<" + Measure Image File : "<<MesureIm<<endl;
    ofs<<" + Finished computation at " << std::ctime(&end_time);
    ofs<<" + Duration: " << elapsed_seconds.count() << "s\n";
    ofs.close();

    ELISE_fp::CpFile(aResRollFile,aDir);
    OutTxtOnlyName = OutTxtOnlyName + "_RollCtrl.txt";
    if (EAMIsInit(&OutRollResTxt))
    {
        cout<<"Init !"<<endl;
        cout<<" Chekc "<<OutTxtOnlyName<<endl;
        if (ELISE_fp::exist_file(OutTxtOnlyName))
        {
            cout<<"File Found !"<<endl;
            ELISE_fp::MvFile(OutTxtOnlyName , OutRollResTxt);
        }
    }
    else
    {
        cout<<"NO Init !"<<endl;
        cout<<" Chekc "<<OutTxtOnlyName<<endl;
        if (ELISE_fp::exist_file(OutTxtOnlyName))
        {
            cout<<"File Found !"<<endl;
            OutRollResTxt = "RollBasc" + OutTxtOnlyName;
            ELISE_fp::MvFile(OutTxtOnlyName , OutRollResTxt);
        }
    }


    std::cout << " + finished computation at " << std::ctime(&end_time)
              << " + Duration: " << elapsed_seconds.count() << "s\n"
              << " + Result Wirtten in: " << OutRollResTxt << "\n";

    ELISE_fp::RmDir(TmpDicoPtsXML);

   if (MMVisualMode) return EXIT_SUCCESS;
   return EXIT_SUCCESS;
}
