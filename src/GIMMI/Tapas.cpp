#include "Tapas.h"

Tapas::Tapas(QList <QString> list, QList <QString> listCheckedItems, QString path) : QWidget()
{



    sendCons="";
    path_s = path.toStdString();
    this->setWindowFlags(Qt::WindowStaysOnTopHint);
    setWindowModality(Qt::ApplicationModal); //IMPORTANT, permet que ta fenêtre reste active

    setWindowTitle("Tapas");
    qDebug()<< path;
    QVBoxLayout *boxlayoutv = new QVBoxLayout;
//    QLabel *labelImages = new QLabel(this);
//    labelImages->setText("Choose the images that you want to use :");
//    labelImages->setStyleSheet("font-weight: bold");

    QLabel *labelMode = new QLabel(this);
    labelMode->setText("Orientation : ");
    labelMode->setStyleSheet("font-weight: bold");

//    QLabel *labelSizeW = new QLabel(this);
//    labelSizeW->setText(" Name that you want to give to your orientation folder : ");
//    labelSizeW->setStyleSheet("font-weight: bold");

    QLabel *labelOptions = new QLabel(this);

    labelOptions->setText("Choose your options");
    labelOptions->setStyleSheet("font-weight: bold");



    QLabel *Out = new QLabel(this);
    Out->setText("Out");
    OutQt = new QTextEdit();
    QHBoxLayout *boxlayouth0 = new QHBoxLayout;
    boxlayouth0->addWidget(Out);
    boxlayouth0->addWidget(OutQt);
    OutQt->setFixedWidth(300);
    OutQt->setFixedHeight(27);

    QLabel *inOri = new QLabel(this);
    inOri->setText("InOri");
    inOriQt = new QTextEdit();
    QHBoxLayout *boxlayouth1 = new QHBoxLayout;
    boxlayouth1->addWidget(inOri);
    boxlayouth1->addWidget(inOriQt);
    inOriQt->setFixedWidth(300);
    inOriQt->setFixedHeight(27);

    QLabel *inCal = new QLabel(this);
    inCal->setText("InCal");
    inCalQt = new QTextEdit();
    QHBoxLayout *boxlayouth2 = new QHBoxLayout;
    boxlayouth2->addWidget(inCal);
    boxlayouth2->addWidget(inCalQt);
    inCalQt->setFixedWidth(300);
    inCalQt->setFixedHeight(27);



//      ExpTxt INT
    QLabel *ExpTxt = new QLabel(this);
    ExpTxt->setText("ExpTxt");
    ExpTxtQt = new QTextEdit();
    QHBoxLayout *boxlayouth3 = new QHBoxLayout;
    boxlayouth3->addWidget(ExpTxt);
    boxlayouth3->addWidget(ExpTxtQt);
    ExpTxtQt->setFixedWidth(300);
    ExpTxtQt->setFixedHeight(27);
//      DoC] INT :: {Do Compensation}
    QLabel *DoC = new QLabel(this);
    DoC->setText("DoC");
    DoCQt = new QTextEdit();
    QHBoxLayout *boxlayouth4 = new QHBoxLayout;
    boxlayouth4->addWidget(DoC);
    boxlayouth4->addWidget(DoCQt);
    DoCQt->setFixedWidth(300);
    DoCQt->setFixedHeight(27);
//      ForCalib] INT :: {Is for calibration (Change def value of LMV and prop diag)?}
    QLabel *ForCalib = new QLabel(this);
    ForCalib->setText("ForCalib");
    ForCalibQt = new QTextEdit();
    QHBoxLayout *boxlayouth5 = new QHBoxLayout;
    boxlayouth5->addWidget(ForCalib);
    boxlayouth5->addWidget(ForCalibQt);
    ForCalibQt->setFixedWidth(300);
    ForCalibQt->setFixedHeight(27);
//      Focs] Pt2dr :: {Keep images with focal length inside range [A,B] (A,B in mm) (Def=keep all)}
    QLabel *Focs = new QLabel(this);
    Focs->setText("Focs");
    FocsQt = new QTextEdit();
    FocsQt2 = new QTextEdit();
    QHBoxLayout *boxlayouth6 = new QHBoxLayout;
    QHBoxLayout *boxlayouth66 = new QHBoxLayout;
    boxlayouth6->addWidget(Focs);
    boxlayouth66->addWidget(FocsQt);
    boxlayouth66->addWidget(FocsQt2);
    FocsQt->setFixedWidth(147);
    FocsQt->setFixedHeight(27);
    FocsQt2->setFixedWidth(147);
    FocsQt2->setFixedHeight(27);
    boxlayouth6->addLayout(boxlayouth66);
//      VitesseInit] INT
    QLabel *VitesseInit = new QLabel(this);
    VitesseInit->setText("VitesseInit");
    VitesseInitQt = new QTextEdit();
    QHBoxLayout *boxlayouth7 = new QHBoxLayout;
    boxlayouth7->addWidget(VitesseInit);
    boxlayouth7->addWidget(VitesseInitQt);
    VitesseInitQt->setFixedWidth(300);
    VitesseInitQt->setFixedHeight(27);
//      PPRel] Pt2dr :: {Principal point shift}
    QLabel *PPRel = new QLabel(this);
    PPRel->setText("PPRel");
    PPRelQt = new QTextEdit();
    PPRelQt2 = new QTextEdit();
    QHBoxLayout *boxlayouth8 = new QHBoxLayout;
    QHBoxLayout *boxlayouth88 = new QHBoxLayout;
    boxlayouth8->addWidget(PPRel);
    boxlayouth8->addWidget(PPRelQt);
    boxlayouth8->addWidget(PPRelQt2);
    PPRelQt->setFixedWidth(147);
    PPRelQt->setFixedHeight(27);
    PPRelQt2->setFixedWidth(147);
    PPRelQt2->setFixedHeight(27);
    boxlayouth8->addLayout(boxlayouth88);
//      Decentre] INT :: {Principal point is shifted (Def=false)}
    QLabel *Decentre = new QLabel(this);
    Decentre->setText("Decentre");
    DecentreQt = new QTextEdit();
    QHBoxLayout *boxlayouth9 = new QHBoxLayout;
    boxlayouth9->addWidget(Decentre);
    boxlayouth9->addWidget(DecentreQt);
    DecentreQt->setFixedWidth(300);
    DecentreQt->setFixedHeight(27);
//      PropDiag] REAL :: {Hemi-spherik fisheye diameter to diagonal ratio}
    QLabel *PropDiag = new QLabel(this);
    PropDiag->setText("PropDiag");
    PropDiagQt = new QTextEdit();
    QHBoxLayout *boxlayouth10 = new QHBoxLayout;
    boxlayouth10->addWidget(PropDiag);
    boxlayouth10->addWidget(PropDiagQt);
    PropDiagQt->setFixedWidth(300);
    PropDiagQt->setFixedHeight(27);
//      SauvAutom] string :: {Save intermediary results to, Set NONE if dont want any}
    QLabel *SauvAutom = new QLabel(this);
    SauvAutom->setText("SauvAutom");
    SauvAutomQt = new QTextEdit();
    QHBoxLayout *boxlayouth11= new QHBoxLayout;
    boxlayouth11->addWidget(SauvAutom);
    boxlayouth11->addWidget(SauvAutomQt);
    SauvAutomQt->setFixedWidth(300);
    SauvAutomQt->setFixedHeight(27);
//      ImInit] string :: {Force first image}
    QLabel *ImInit = new QLabel(this);
    ImInit->setText("ImInit");
    ImInitQt = new QTextEdit();
    QHBoxLayout *boxlayouth12= new QHBoxLayout;
    boxlayouth12->addWidget(ImInit);
    boxlayouth12->addWidget(ImInitQt);
    ImInitQt->setFixedWidth(300);
    ImInitQt->setFixedHeight(27);
//      MOI] bool :: {MOI}
    QLabel *MOI = new QLabel(this);
    MOI->setText("MOI");
    MOIQt = new QCheckBox();
    QHBoxLayout *boxlayouth13= new QHBoxLayout;
    boxlayouth13->addWidget(MOI);
    boxlayouth13->addWidget(MOIQt);
    MOIQt->setFixedWidth(300);
    MOIQt->setFixedHeight(27);
//      DBF] INT :: {Debug (internal use : DebugPbCondFaisceau=true) }
    QLabel *DBF = new QLabel(this);
    DBF->setText("DBF");
    DBFQt = new QTextEdit();
    QHBoxLayout *boxlayouth14 = new QHBoxLayout;
    boxlayouth14->addWidget(DBF);
    boxlayouth14->addWidget(DBFQt);
    DBFQt->setFixedWidth(300);
    DBFQt->setFixedHeight(27);
//      Debug] bool :: {Partial file for debug}
    QLabel *Debug = new QLabel(this);
    Debug->setText("Debug");
    DebugQt = new QCheckBox();
    QHBoxLayout *boxlayouth15 = new QHBoxLayout;
    boxlayouth15->addWidget(Debug);
    boxlayouth15->addWidget(DebugQt);
    DebugQt->setFixedWidth(300);
    DebugQt->setFixedHeight(27);
//      DegRadMax] INT :: {Max degree of radial, default model dependent}
    QLabel *DegRadMax = new QLabel(this);
    DegRadMax->setText("DegRadMax");
    DegRadMaxQt = new QTextEdit();
    QHBoxLayout *boxlayouth16 = new QHBoxLayout;
    boxlayouth16->addWidget(DegRadMax);
    boxlayouth16->addWidget(DegRadMaxQt);
    DegRadMaxQt->setFixedWidth(300);
    DegRadMaxQt->setFixedHeight(27);
//      DegGen] INT :: {Max degree of general polynome, default model dependent (generally 0 or 1)}
    QLabel *DegGen = new QLabel(this);
    DegGen->setText("DegGen");
    DegGenQt = new QTextEdit();
    QHBoxLayout *boxlayouth17= new QHBoxLayout;
    boxlayouth17->addWidget(DegGen);
    boxlayouth17->addWidget(DegGenQt);
    DegGenQt->setFixedWidth(300);
    DegGenQt->setFixedHeight(27);
//      LibAff] bool :: {Free affine parameter, Def=true}
    QLabel *LibAff = new QLabel(this);
    LibAff->setText("LibAff");
    LibAffQt = new QCheckBox();
    QHBoxLayout *boxlayouth18= new QHBoxLayout;
    boxlayouth18->addWidget(LibAff);
    boxlayouth18->addWidget(LibAffQt);
    LibAffQt->setFixedWidth(300);
    LibAffQt->setFixedHeight(27);
//      LibDec] bool :: {Free decentric parameter, Def=true}
    QLabel *LibDec = new QLabel(this);
    LibDec->setText("LibDec");
    LibDecQt = new QCheckBox();
    QHBoxLayout *boxlayouth19= new QHBoxLayout;
    boxlayouth19->addWidget(LibDec);
    boxlayouth19->addWidget(LibDecQt);
    LibDecQt->setFixedWidth(300);
    LibDecQt->setFixedHeight(27);
//      LibPP] bool :: {Free principal point, Def=true}
    QLabel *LibPP = new QLabel(this);
    LibPP->setText("LibPP");
    LibPPQt = new QCheckBox();
    QHBoxLayout *boxlayouth20= new QHBoxLayout;
    boxlayouth20->addWidget(LibPP);
    boxlayouth20->addWidget(LibPPQt);
    LibPPQt->setFixedWidth(300);
    LibPPQt->setFixedHeight(27);
//      LibCP] bool :: {Free distorsion center, Def=true}
    QLabel *LibCP = new QLabel(this);
    LibCP->setText("LibCP");
    LibCPQt = new QCheckBox();
    QHBoxLayout *boxlayouth21= new QHBoxLayout;
    boxlayouth21->addWidget(LibCP);
    boxlayouth21->addWidget(LibCPQt);
    LibCPQt->setFixedWidth(300);
    LibCPQt->setFixedHeight(27);
//      LibFoc] bool :: {Free focal, Def=true}
    QLabel *LibFoc = new QLabel(this);
    LibFoc->setText("LibFoc");
    LibFocQt = new QCheckBox();
    QHBoxLayout *boxlayouth22= new QHBoxLayout;
    boxlayouth22->addWidget(LibFoc);
    boxlayouth22->addWidget(LibFocQt);
    LibFocQt->setFixedWidth(300);
    LibFocQt->setFixedHeight(27);
//      RapTxt] string :: {RapTxt}
    QLabel *RapTxt = new QLabel(this);
    RapTxt->setText("RapTxt");
    RapTxtQt = new QTextEdit();
    QHBoxLayout *boxlayouth23= new QHBoxLayout;
    boxlayouth23->addWidget(RapTxt);
    boxlayouth23->addWidget(RapTxtQt);
    RapTxtQt->setFixedWidth(300);
    RapTxtQt->setFixedHeight(27);
//      LinkPPaPPs] REAL :: {Link PPa and PPs (double)}
    QLabel *LinkPPaPPs = new QLabel(this);
    LinkPPaPPs->setText("LinkPPaPPs");
    LinkPPaPPsQt = new QTextEdit();
    QHBoxLayout *boxlayouth24 = new QHBoxLayout;
    boxlayouth24->addWidget(LinkPPaPPs);
    boxlayouth24->addWidget(LinkPPaPPsQt);
    LinkPPaPPsQt->setFixedWidth(300);
    LinkPPaPPsQt->setFixedHeight(27);
//      FrozenPoses] string :: {List of frozen poses (pattern)}
    QLabel *FrozenPoses = new QLabel(this);
    FrozenPoses->setText("FrozenPoses");
    FrozenPosesQt = new QTextEdit();
    QHBoxLayout *boxlayouth25= new QHBoxLayout;
    boxlayouth25->addWidget(FrozenPoses);
    boxlayouth25->addWidget(FrozenPosesQt);
    FrozenPosesQt->setFixedWidth(300);
    FrozenPosesQt->setFixedHeight(27);
//      FrozenCenters] string :: {List of frozen centers of poses (pattern)}
    QLabel *FrozenCenters = new QLabel(this);
    FrozenCenters->setText("FrozenCenters");
    FrozenCentersQt = new QTextEdit();
    QHBoxLayout *boxlayouth26= new QHBoxLayout;
    boxlayouth26->addWidget(FrozenCenters);
    boxlayouth26->addWidget(FrozenCentersQt);
    FrozenCentersQt->setFixedWidth(300);
    FrozenCentersQt->setFixedHeight(27);
//      FrozenOrients] string :: {List of frozen orients of poses (pattern)}
    QLabel *FrozenOrients = new QLabel(this);
    FrozenOrients->setText("FrozenOrients");
    FrozenOrientsQt = new QTextEdit();
    QHBoxLayout *boxlayouth27= new QHBoxLayout;
    boxlayouth27->addWidget(FrozenOrients);
    boxlayouth27->addWidget(FrozenOrientsQt);
    FrozenOrientsQt->setFixedWidth(300);
    FrozenOrientsQt->setFixedHeight(27);
//      FreeCalibInit] bool :: {Free calibs as soon as created (Def=false)}
    QLabel *FreeCalibInit = new QLabel(this);
    FreeCalibInit->setText("FreeCalibInit");
    FreeCalibInitQt = new QCheckBox();
    QHBoxLayout *boxlayouth28= new QHBoxLayout;
    boxlayouth28->addWidget(FreeCalibInit);
    boxlayouth28->addWidget(FreeCalibInitQt);
    FreeCalibInitQt->setFixedWidth(300);
    FreeCalibInitQt->setFixedHeight(27);
//      FrozenCalibs] string :: {List of frozen calibration (pattern)}
    QLabel *FrozenCalibs = new QLabel(this);
    FrozenCalibs->setText("FrozenCalibs");
    FrozenCalibsQt = new QTextEdit();
    QHBoxLayout *boxlayouth29= new QHBoxLayout;
    boxlayouth29->addWidget(FrozenCalibs);
    boxlayouth29->addWidget(FrozenCalibsQt);
    FrozenCalibsQt->setFixedWidth(300);
    FrozenCalibsQt->setFixedHeight(27);
//      FreeCalibs] string :: {List of free calibration (pattern, Def=".*")}
    QLabel *FreeCalibs = new QLabel(this);
    FreeCalibs->setText("FreeCalibs");
    FreeCalibsQt = new QTextEdit();
    QHBoxLayout *boxlayouth30= new QHBoxLayout;
    boxlayouth30->addWidget(FreeCalibs);
    boxlayouth30->addWidget(FreeCalibsQt);
    FreeCalibsQt->setFixedWidth(300);
    FreeCalibsQt->setFixedHeight(27);
//      SH] string :: {Set of Hom, Def="", give MasqFiltered for result of HomolFilterMasq}
    QLabel *SH = new QLabel(this);
    SH->setText("SH");
    SHQt = new QTextEdit();
    QHBoxLayout *boxlayouth31= new QHBoxLayout;
    boxlayouth31->addWidget(SH);
    boxlayouth31->addWidget(SHQt);
    SHQt->setFixedWidth(300);
    SHQt->setFixedHeight(27);
//      RefineAll] bool :: {More refinement at all step, safer and more accurate, but slower, def=true}
    QLabel *RefineAll = new QLabel(this);
    RefineAll->setText("RefineAll");
    RefineAllQt = new QCheckBox();
    QHBoxLayout *boxlayouth32= new QHBoxLayout;
    boxlayouth32->addWidget(RefineAll);
    boxlayouth32->addWidget(RefineAllQt);
    RefineAllQt->setFixedWidth(300);
    RefineAllQt->setFixedHeight(27);
//      ImMinMax] vector<std::string> :: {Image min and max (may avoid tricky pattern ...)}
    QLabel *ImMinMax = new QLabel(this);
    ImMinMax->setText("ImMinMax");
    ImMinMaxQt = new QTextEdit();
    QHBoxLayout *boxlayouth33= new QHBoxLayout;
    boxlayouth33->addWidget(ImMinMax);
    boxlayouth33->addWidget(ImMinMaxQt);
    ImMinMaxQt->setFixedWidth(300);
    ImMinMaxQt->setFixedHeight(27);
//      EcMax] REAL :: {Final threshold for residual, def = 5.0 }
    QLabel *EcMax = new QLabel(this);
    EcMax->setText("EcMax");
    EcMaxQt = new QTextEdit();
    QHBoxLayout *boxlayouth34= new QHBoxLayout;
    boxlayouth34->addWidget(EcMax);
    boxlayouth34->addWidget(EcMaxQt);
    EcMaxQt->setFixedWidth(300);
    EcMaxQt->setFixedHeight(27);
//      EcInit] Pt2dr :: {Inital threshold for residual def = [100,5.0] }
    QLabel *EcInit = new QLabel(this);
    EcInit->setText("EcInit");
    EcInitQt = new QTextEdit();
    EcInitQt2 = new QTextEdit();
    QHBoxLayout *boxlayouth35= new QHBoxLayout;
    QHBoxLayout *boxlayouth355= new QHBoxLayout;
    boxlayouth35->addWidget(EcInit);
    boxlayouth355->addWidget(EcInitQt);
    boxlayouth355->addWidget(EcInitQt2);
    EcInitQt->setFixedWidth(147);
    EcInitQt->setFixedHeight(27);
    EcInitQt2->setFixedWidth(147);
    EcInitQt2->setFixedHeight(27);
    boxlayouth35->addLayout(boxlayouth355);
//      CondMaxPano] REAL :: {Precaution for conditionning with Panoramic images, Def=1e4 (old was 0) }
    QLabel *CondMaxPano = new QLabel(this);
    CondMaxPano->setText("CondMaxPano");
    CondMaxPanoQt = new QTextEdit();
    QHBoxLayout *boxlayouth36= new QHBoxLayout;
    boxlayouth36->addWidget(CondMaxPano);
    boxlayouth36->addWidget(CondMaxPanoQt);
    CondMaxPanoQt->setFixedWidth(300);
    CondMaxPanoQt->setFixedHeight(27);
//      SinglePos] vector<std::string> :: {Pattern of single Pos Calib to save [Pose,Calib]}
    QLabel *SinglePos = new QLabel(this);
    SinglePos->setText("SinglePos");
    SinglePosQt = new QTextEdit();
    QHBoxLayout *boxlayouth37= new QHBoxLayout;
    boxlayouth37->addWidget(SinglePos);
    boxlayouth37->addWidget(SinglePosQt);
    SinglePosQt->setFixedWidth(300);
    SinglePosQt->setFixedHeight(27);
//      RankInitF] INT :: {Order of focal initialisation, ref id distotion =2, Def=3 }
    QLabel *RankInitF = new QLabel(this);
    RankInitF->setText("RankInitF");
    RankInitFQt = new QTextEdit();
    QHBoxLayout *boxlayouth38= new QHBoxLayout;
    boxlayouth38->addWidget(RankInitF);
    boxlayouth38->addWidget(RankInitFQt);
    RankInitFQt->setFixedWidth(300);
    RankInitFQt->setFixedHeight(27);
//      RankInitPP] INT :: {Order of Principal point initialisation, ref id distotion =2, Def=4}
    QLabel *RankInitPP = new QLabel(this);
    RankInitPP->setText("RankInitPP");
    RankInitPPQt = new QTextEdit();
    QHBoxLayout *boxlayouth39= new QHBoxLayout;
    boxlayouth39->addWidget(RankInitPP);
    boxlayouth39->addWidget(RankInitPPQt);
    RankInitPPQt->setFixedWidth(300);
    RankInitPPQt->setFixedHeight(27);
//      RegulDist] vector<double> :: {Parameter fo RegulDist [Val,Grad,Hessian,NbCase,SeuilNb]}
    QLabel *RegulDist = new QLabel(this);
    RegulDist->setText("RegulDist");
    RegulDistQt = new QTextEdit();
    QHBoxLayout *boxlayouth40= new QHBoxLayout;
    boxlayouth40->addWidget(RegulDist);
    boxlayouth40->addWidget(RegulDistQt);
    RegulDistQt->setFixedWidth(300);
    RegulDistQt->setFixedHeight(27);
//      MulLVM] REAL :: {Multipier Levenberg Markard}
    QLabel *MulLVM = new QLabel(this);
    MulLVM->setText("MulLVM");
    MulLVMQt = new QTextEdit();
    QHBoxLayout *boxlayouth41= new QHBoxLayout;
    boxlayouth41->addWidget(MulLVM);
    boxlayouth41->addWidget(MulLVMQt);
    MulLVMQt->setFixedWidth(300);
    MulLVMQt->setFixedHeight(27);
//      MultipleBlock] bool :: {Multiple block need special caution (only related to Levenberg Markard)}
    QLabel *MultipleBlock = new QLabel(this);
    MultipleBlock->setText("MultipleBlock");
    MultipleBlockQt = new QCheckBox();
    QHBoxLayout *boxlayouth42 = new QHBoxLayout;
    boxlayouth42->addWidget(MultipleBlock);
    boxlayouth42->addWidget(MultipleBlockQt);
    MultipleBlockQt->setFixedWidth(300);
    MultipleBlockQt->setFixedHeight(27);











//    boxlayoutv->addWidget(labelImages);


    for (int i = 0; i < list.count(); i++)
    {
        std::string mot = list[i].toStdString();
        if((mot.find(".JPG")<10000) ||
                (mot.find(".jpg")<10000) ||
                (mot.find(".JPEG")<10000) ||
                (mot.find(".jpeg")<10000) ||
                (mot.find(".png")<10000) ||
                (mot.find(".PNG")<10000) ||
                (mot.find(".bmp")<10000) ||
                (mot.find(".BMP")<10000) ||
                (mot.find(".TIF")<10000) ||
                (mot.find(".tif")<10000)){
            listImages.append(list[i]);
     }
    }

    for (int i = 0; i < listImages.count(); i++)
    {
        imagesCheckBox = new QCheckBox(listImages[i]);
        listeCasesImages.append(imagesCheckBox);

        for (int y = 0; y < listImages.count(); y++)
            if(listImages[i]==listCheckedItems[y]){
                qDebug() << listImages[i] + "      listImages";
                qDebug() << listCheckedItems[i] + "listCheckedItems";
                imagesCheckBox->setChecked(true);
                y=listImages.count();
            }
    }



    groupBox = new QComboBox();
    groupBox->addItem("RadialBasic");
    groupBox->addItem("RadialExtended");
    groupBox->addItem("Fraser");
    groupBox->addItem("FishEyeEqui");
    groupBox->addItem("AutoCal");
    groupBox->addItem("Figee");
    groupBox->addItem("HemiEqui");
    groupBox->addItem("RadialStd");
    groupBox->addItem("FraserBasic");
    groupBox->addItem("FishEyeBasic");
    groupBox->addItem("FE_EquiSolBasic");
    groupBox->addItem("Four7x2");
    groupBox->addItem("Four11x2");
    groupBox->addItem("Four15x2");
    groupBox->addItem("Four19x2");
    groupBox->addItem("AddFour7x2");
    groupBox->addItem("AddFour11x2");
    groupBox->addItem("AddFour15x2");
    groupBox->addItem("AddFour19x2");
    groupBox->addItem("AddPolyDeg0");
    groupBox->addItem("AddPolyDeg1");
    groupBox->addItem("AddPolyDeg2");
    groupBox->addItem("AddPolyDeg3");
    groupBox->addItem("AddPolyDeg4");
    groupBox->addItem("AddPolyDeg5");
    groupBox->addItem("AddPolyDeg6");
    groupBox->addItem("AddPolyDeg7");
    groupBox->addItem("Ebner");
    groupBox->addItem("Brown");


    boxlayoutv->addWidget(labelMode);
    boxlayoutv->addWidget(groupBox);





//    boxlayoutv->addWidget(labelSizeW);
//    outS = new QTextEdit();

//    outS->setFixedHeight(27);

//    boxlayoutv->addWidget(outS);



   boxlayoutv->addWidget(labelOptions);
//    boxTxt = new QCheckBox("Txt");
//    boxlayoutv->addWidget(boxTxt);

   QScrollArea *scrollArea = new QScrollArea;
   scrollArea->setFixedSize(440, 300);
   QWidget *layoutWidget = new QWidget(this);
   QVBoxLayout *boxlayoutscroll = new QVBoxLayout;
   layoutWidget->setLayout(boxlayoutscroll);

    boxlayoutscroll->addLayout(boxlayouth0);
    boxlayoutscroll->addLayout(boxlayouth1);
    boxlayoutscroll->addLayout(boxlayouth2);
    boxlayoutscroll->addLayout(boxlayouth3);
    boxlayoutscroll->addLayout(boxlayouth4);
    boxlayoutscroll->addLayout(boxlayouth5);
    boxlayoutscroll->addLayout(boxlayouth6);
    boxlayoutscroll->addLayout(boxlayouth7);
    boxlayoutscroll->addLayout(boxlayouth8);
    boxlayoutscroll->addLayout(boxlayouth9);
    boxlayoutscroll->addLayout(boxlayouth10);
    boxlayoutscroll->addLayout(boxlayouth11);
    boxlayoutscroll->addLayout(boxlayouth12);
    boxlayoutscroll->addLayout(boxlayouth13);
    boxlayoutscroll->addLayout(boxlayouth14);
    boxlayoutscroll->addLayout(boxlayouth15);
    boxlayoutscroll->addLayout(boxlayouth16);
    boxlayoutscroll->addLayout(boxlayouth17);
    boxlayoutscroll->addLayout(boxlayouth18);
    boxlayoutscroll->addLayout(boxlayouth19);
    boxlayoutscroll->addLayout(boxlayouth20);
    boxlayoutscroll->addLayout(boxlayouth21);
    boxlayoutscroll->addLayout(boxlayouth22);
    boxlayoutscroll->addLayout(boxlayouth23);
    boxlayoutscroll->addLayout(boxlayouth24);
    boxlayoutscroll->addLayout(boxlayouth25);
    boxlayoutscroll->addLayout(boxlayouth26);
    boxlayoutscroll->addLayout(boxlayouth27);
    boxlayoutscroll->addLayout(boxlayouth28);
    boxlayoutscroll->addLayout(boxlayouth29);
    boxlayoutscroll->addLayout(boxlayouth30);
    boxlayoutscroll->addLayout(boxlayouth31);
    boxlayoutscroll->addLayout(boxlayouth32);
    boxlayoutscroll->addLayout(boxlayouth33);
    boxlayoutscroll->addLayout(boxlayouth34);
    boxlayoutscroll->addLayout(boxlayouth35);
    boxlayoutscroll->addLayout(boxlayouth36);
    boxlayoutscroll->addLayout(boxlayouth37);
    boxlayoutscroll->addLayout(boxlayouth38);
    boxlayoutscroll->addLayout(boxlayouth39);
    boxlayoutscroll->addLayout(boxlayouth40);
    boxlayoutscroll->addLayout(boxlayouth41);
    boxlayoutscroll->addLayout(boxlayouth42);


    boxlayoutv->addWidget(scrollArea);


    scrollArea->setWidget(layoutWidget);
    scrollArea->show();

    QPushButton *ok = new QPushButton();
    ok->setText("Ok");
    ok->setFixedWidth(70);
    ok->setFixedHeight(40);



    boxlayoutv->addWidget(ok);




    QObject::connect(ok, SIGNAL(clicked()), this, SLOT(mm3d()));

    this->setLayout(boxlayoutv);





}

void Tapas::mm3d(){


    int a = 0;
    images.erase();

    std::cout <<  "images = "  << std::endl;

    for( int i = 0; i< listeCasesImages.size(); i++){
        if(listeCasesImages.at(i)->isChecked())
        {a+=1;
            images.append(listImages.at(i).toStdString()+"|");



        }
    }

    if(a<2){
        msgBox2.setWindowTitle("Warning");

        msgBox2.setText("Veuillez sélectionner au moins deux images à traiter \n ");
        msgBox2.setStandardButtons(QMessageBox::Ok);
        msgBox2.setDefaultButton(QMessageBox::Ok);
        if(msgBox2.exec() == QMessageBox::Ok){



        }




    }else{

        images = images.erase(images.size()-1);
        std::cout <<  images << std::endl;


        mode = groupBox->currentText().toStdString();

        std::cout <<  "mode = " + mode << std::endl;

//        QTextEdit
        if(OutQt->toPlainText()!=""){
            OutVar="Out="+OutQt->toPlainText().toStdString();
        }
//        QTextEdit
        if(inOriQt->toPlainText()!=""){
            inOriVar="inOri="+inOriQt->toPlainText().toStdString();
        }
//        QTextEdit
        if(inCalQt->toPlainText()!=""){
            inCalVar="inCal="+inCalQt->toPlainText().toStdString();
        }
//        QTextEdit
        if(ExpTxtQt->toPlainText()!=""){
            ExpTxtVar="ExpTxt="+ExpTxtQt->toPlainText().toStdString();
        }
//        QTextEdit
        if(DoCQt->toPlainText()!=""){
            DoCVar="DoC="+DoCQt->toPlainText().toStdString();
        }
//        QTextEdit
        if(ForCalibQt->toPlainText()!=""){
            ForCalibVar="ForCalib="+ForCalibQt->toPlainText().toStdString();
        }

       if(FocsQt->toPlainText()!="" && FocsQt2->toPlainText()!=""){
           FocsVar="Focs=["+FocsQt->toPlainText().toStdString()+";"+FocsQt2->toPlainText().toStdString()+"]";
       }

//        QTextEdit *
       if( VitesseInitQt->toPlainText()!=""){
            VitesseInitVar="VitesseInit="+VitesseInitQt->toPlainText().toStdString();
        }


        if(PPRelQt->toPlainText()!="" && PPRelQt2->toPlainText()!=""){
            PPRelVar="PPRel=["+PPRelQt->toPlainText().toStdString()+";"+PPRelQt2->toPlainText().toStdString()+"]";
        }

//        QTextEdit *
        if(DecentreQt->toPlainText()!=""){
            DecentreVar="Decentre="+DecentreQt->toPlainText().toStdString();
        }
//        QTextEdit *
        if(PropDiagQt->toPlainText()!=""){
            PropDiagVar="PropDiag="+PropDiagQt->toPlainText().toStdString();
        }
//        QTextEdit *
        if(SauvAutomQt->toPlainText()!=""){
            SauvAutomVar="SauvAutom="+SauvAutomQt->toPlainText().toStdString();
        }
//        QTextEdit *
        if(ImInitQt->toPlainText()!=""){
            ImInitVar="ImInit="+ImInitQt->toPlainText().toStdString();
        }
//        QCheckBox *
        if(MOIQt->isChecked()){
            MOIVar="MOI=1";
        }
//        QTextEdit *
        if(DBFQt->toPlainText()!=""){
            DBFVar="DBF="+DBFQt->toPlainText().toStdString();
        }
//        QCheckBox *
        if(DebugQt->isChecked()){
                DebugVar="Debug=1";
            }
//        QTextEdit *
        if(DegRadMaxQt->toPlainText()!=""){
            DegRadMaxVar="DegRadMax="+DegRadMaxQt->toPlainText().toStdString();
        }
//        QTextEdit *
        if(DegGenQt->toPlainText()!=""){
            DegGenVar="DegGen="+DegGenQt->toPlainText().toStdString();
        }
//        QCheckBox *
        if(LibAffQt->isChecked()){
            LibAffVar="LibAff=1";
        }
//        QCheckBox *
        if(LibDecQt->isChecked()){
            LibDecVar="LibDec=1";
        }
//      if(  QCheckBox
        if(LibPPQt->isChecked()){
            LibPPVar="LibPP=1";
        }
//        QCheckBox
        if(LibCPQt->isChecked()){
            LibCPVar="LibCP=1";
        }
//        QCheckBox
       if( LibFocQt->isChecked()){
           LibFocVar="LibFoc=1";
       }
//        QTextEdit
        if(RapTxtQt->toPlainText()!=""){
            RapTxtVar="RapTxt="+RapTxtQt->toPlainText().toStdString();
        }
//        QTextEdit
        if(LinkPPaPPsQt->toPlainText()!=""){
            LinkPPaPPsVar="LinkPPaPPs="+LinkPPaPPsQt->toPlainText().toStdString();
        }
//        QTextEdit
        if(FrozenPosesQt->toPlainText()!=""){
            FrozenPosesVar="FrozenPoses="+FrozenPosesQt->toPlainText().toStdString();
        }
//        QTextEdit
       if( FrozenCentersQt->toPlainText()!=""){
            FrozenCentersVar="FrozenCenters="+FrozenCentersQt->toPlainText().toStdString();
        }
//        QTextEdit
        if(FrozenOrientsQt->toPlainText()!=""){
            FrozenOrientsVar="FrozenOrients="+FrozenOrientsQt->toPlainText().toStdString();
        }
//        QCheckBox
        if(FreeCalibInitQt->isChecked()){
            FreeCalibInitVar="FreeCalibInit=1";
        }
//        QTextEdit
        if(FrozenCalibsQt->toPlainText()!=""){
            FrozenCalibsVar="FrozenCalibs="+FrozenCalibsQt->toPlainText().toStdString();
        }
//        QTextEdit
       if( FreeCalibsQt->toPlainText()!=""){
            FreeCalibsVar="FreeCalibs="+FreeCalibsQt->toPlainText().toStdString();
        }
//        QTextEdit
        if(SHQt->toPlainText()!=""){
            SHVar="SH="+SHQt->toPlainText().toStdString();
        }
//        QCheckBox
        if(RefineAllQt->isChecked()){
            RefineAllVar="RefineAll=1";
        }
//        QTextEdit
        if(ImMinMaxQt->toPlainText()!=""){
            ImMinMaxVar="ImMinMax="+ImMinMaxQt->toPlainText().toStdString();
        }
//        QTextEdit
        if(EcMaxQt->toPlainText()!=""){
            EcMaxVar="EcMax="+EcMaxQt->toPlainText().toStdString();
        }
//        QTextEdit
        if(EcInitQt->toPlainText()!="" && EcInitQt2->toPlainText()!=""){
            EcInitVar="EcInit=["+EcInitQt->toPlainText().toStdString()+";"+EcInitQt2->toPlainText().toStdString()+"]";
        }
//        QTextEdit
        if(CondMaxPanoQt->toPlainText()!=""){
            CondMaxPanoVar="CondMaxPano="+CondMaxPanoQt->toPlainText().toStdString();
        }
//        QTextEdit
        if(SinglePosQt->toPlainText()!=""){
            SinglePosVar="SinglePos="+SinglePosQt->toPlainText().toStdString();
        }
//        QTextEdit
        if(RankInitFQt->toPlainText()!=""){
            RankInitFVar="RankInitF="+RankInitFQt->toPlainText().toStdString();
        }
//        QTextEdit
        if(RankInitPPQt->toPlainText()!=""){
            RankInitPPVar="RankInitPP="+RankInitPPQt->toPlainText().toStdString();
        }
//        QTextEdit
        if(RegulDistQt->toPlainText()!=""){
            RegulDistVar="RegulDist="+RegulDistQt->toPlainText().toStdString();
        }
//        QTextEdit
        if(MulLVMQt->toPlainText()!=""){
            MulLVMVar="MulLVM="+MulLVMQt->toPlainText().toStdString();
        }
//        QCheckBox
        if(MultipleBlockQt->isChecked()){
           MultipleBlockVar="MultipleBlock=1";
       }



        cmd = "mm3d Tapas " + mode +" \""+ images +"\" "+" "
                   + OutVar.c_str()+" "+
                   + inOriVar.c_str()+" "+
                   + inCalVar.c_str()+" "+
                   + ExpTxtVar.c_str()+" "+
                   + DoCVar.c_str()+" "+
                   + ForCalibVar.c_str()+" "+
                   + FocsVar.c_str()+" "+
                   + VitesseInitVar.c_str()+" "+
                   + PPRelVar.c_str()+" "+
                   + DecentreVar.c_str()+" "+
                   + PropDiagVar.c_str()+" "+
                   + SauvAutomVar.c_str()+" "+
                   + ImInitVar.c_str()+" "+
                   + MOIVar.c_str()+" "+
                   + DBFVar.c_str()+" "+
                   + DebugVar.c_str()+" "+
                   + DegRadMaxVar.c_str()+" "+
                   + DegGenVar.c_str()+" "+
                   + LibAffVar.c_str()+" "+
                   + LibDecVar.c_str()+" "+
                   + LibPPVar.c_str()+" "+
                   + LibCPVar.c_str()+" "+
                   + LibFocVar.c_str()+" "+
                   + RapTxtVar.c_str()+" "+
                   + LinkPPaPPsVar.c_str()+" "+
                   + FrozenPosesVar.c_str()+" "+
                   + FrozenCentersVar.c_str()+" "+
                   + FrozenOrientsVar.c_str()+" "+
                   + FreeCalibInitVar.c_str()+" "+
                   + FrozenCalibsVar.c_str()+" "+
                   + FreeCalibsVar.c_str()+" "+
                   + SHVar.c_str()+" "+
                   + RefineAllVar.c_str()+" "+
                   + ImMinMaxVar.c_str()+" "+
                   + EcMaxVar.c_str()+" "+
                   + EcInitVar.c_str()+" "+
                   + CondMaxPanoVar.c_str()+" "+
                   + SinglePosVar.c_str()+" "+
                   + RankInitFVar.c_str()+" "+
                   + RankInitPPVar.c_str()+" "+
                   + RegulDistVar.c_str()+" "+
                   + MulLVMVar.c_str()+" "+
                   + MultipleBlockVar.c_str()+" "+
                                +" @ExitOnBrkp";





        QMessageBox msgBox2;
        msgBox2.setWindowTitle("Tapas");
        const QString str = QString::fromStdString(cmd);
        msgBox2.setText("Here is the commande you're about to launch : \n " + str);
        msgBox2.setStandardButtons(QMessageBox::Yes);
        msgBox2.addButton(QMessageBox::No);
        msgBox2.setDefaultButton(QMessageBox::No);
        if(msgBox2.exec() == QMessageBox::Yes){

            this->close();
            p.setWorkingDirectory(path_s.c_str());
            p.waitForFinished(-1);
            qDebug() <<  QDir::currentPath();
            qDebug() <<  cmd.c_str() ;
            p.start(cmd.c_str());
            p.setReadChannel(QProcess::StandardOutput);
            connect(&p,SIGNAL(readyReadStandardOutput()),this,SLOT(readyReadStandardOutput()));
            connect(&p, SIGNAL(finished(int,QProcess::ExitStatus)), this, SLOT(stopped()));  //Add this line will cause error at runtime
            msg();




        }else{                  images.clear();

                                OutVar=" ";

                                inOriVar=" ";

                                inCalVar=" ";

                                ExpTxtVar=" ";

                                DoCVar=" ";

                                ForCalibVar=" ";

                                FocsVar=" ";

                                Focs2Var=" ";

                                VitesseInitVar=" ";

                                PPRelVar=" ";

                                PPRel2Var=" ";

                                DecentreVar=" ";

                                PropDiagVar=" ";

                                SauvAutomVar=" ";

                                ImInitVar=" ";

                                MOIVar=" ";

                                DBFVar=" ";

                                DebugVar=" ";

                                DegRadMaxVar=" ";

                                DegGenVar=" ";

                                LibAffVar=" ";

                                LibDecVar=" ";

                                LibPPVar=" ";

                                LibCPVar=" ";

                                LibFocVar=" ";

                                RapTxtVar=" ";

                                LinkPPaPPsVar=" ";

                                FrozenPosesVar=" ";

                                FrozenCentersVar=" ";

                                FrozenOrientsVar=" ";

                                FreeCalibInitVar=" ";

                                FrozenCalibsVar=" ";

                                FreeCalibsVar=" ";

                                SHVar=" ";

                                RefineAllVar=" ";

                                ImMinMaxVar=" ";

                                EcMaxVar=" ";

                                EcInitVar=" ";

                                EcInit2Var=" ";

                                CondMaxPanoVar=" ";

                                SinglePosVar=" ";

                                RankInitFVar=" ";

                                RankInitPPVar=" ";

                                RegulDistVar=" ";
                                MulLVMVar=" ";
                                MultipleBlockVar=" ";

        }

        // do something else



    }}

QString Tapas::getTxt(){
    return p_stdout;
}

void Tapas::msg(){

    qDi.setWindowTitle("Tapas state");
    QLabel *wait = new QLabel(this);
    wait->setText(" Wait until the end of the process, or click stop to cancel it ");
    wait->setStyleSheet("font-weight: bold");
    Qtr = new QTextEdit();
    Qtr->setFixedHeight(500);
    QVBoxLayout *QV = new QVBoxLayout();
    QV->addWidget(wait);
    QV->addWidget(Qtr);
    QPushButton *stop = new QPushButton();
    stop->setText("STOP");
    stop->setFixedHeight(30);
    stop->setFixedWidth(60);

                                                                                                                                                                                                        QObject::connect(stop, SIGNAL(clicked()), this, SLOT(stopped()));
                                                                                                                                                                                                                      QV->addWidget(stop);
                                                                                                                                                                                                                                    qDi.setLayout(QV);
                                                                                                                                                                                                                                                  qDi.show();}

QString Tapas::msg1(){
    return sendCons;
}

void Tapas::stopped(){
    sendCons= Qtr->toPlainText();
    qDi.accept();
    this->close();


}

void Tapas::readyReadStandardOutput(){
    Qtr->append(QString::fromStdString( p.readAllStandardOutput().data()));

}

std::string Tapas::sendOri(){
   return mode;
}








