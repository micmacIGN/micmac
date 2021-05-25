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
#include "InitOriLinear.h"
#include "StdAfx.h"

bool checkNumParam(vector<string>aVecPatternNewImages_E , vector<string>aVecPatternRefImages_E ,
                vector<string>aPatPoseTurn , vector<double>aPatAngle, vector<double>aMulF)
{
    bool ok;
    if (aVecPatternNewImages_E.size() == aVecPatternRefImages_E.size())
    {
        cout<<"Number of pattern new image and pattern reference image is good (same)"<<endl;
        ok = true;
        if (aPatPoseTurn.size() > 0 || aPatAngle.size() > 0)
        {
            if (aPatPoseTurn.size() == aPatAngle.size())
            {
                cout<<"Number of pose turn and angle is good (same)"<<endl;
                ok = true;
            }
            else
            {
                cout<<"ERROR : Number of pose turn and angle is NOT the same"<<endl;
                cout<<"ERROR - quit - goodbye !"<<endl;
                ok = false;
            }
            cout<<" ++ Pose turn: "<<aPatPoseTurn.size()<<" poses"<<endl;
            for (uint i=0; i<aPatPoseTurn.size(); i++)
                cout<<" ++ "<<aPatPoseTurn[i]<<endl;
            cout<<" ++ Angle: "<<aPatAngle.size()<<" angls"<<endl;
            for (uint i=0; i<aPatAngle.size(); i++)
                cout<<" ++ "<<aPatAngle[i]<<endl;
        }
    }
    else
    {
        cout<<"ERROR : Number of pattern new image and pattern reference image is NOT the same"<<endl;
        cout<<"ERROR - quit - goodbye !"<<endl;
        ok = false;
    }
    cout<<" ++ Pattern of new images: "<<aVecPatternNewImages_E.size()<<" pats"<<endl;
    for (uint i=0; i<aVecPatternNewImages_E.size(); i++)
        cout<<"     ++ "<<aVecPatternNewImages_E[i]<<endl;
    cout<<" ++ Pattern of Ref images: "<<aVecPatternRefImages_E.size()<<" pats"<<endl;
    for (uint i=0; i<aVecPatternRefImages_E.size(); i++)
        cout<<"     ++ "<<aVecPatternRefImages_E[i]<<endl;
    return ok;
}

vector<string> parseParam(string aParam)
{
    cout<<"Parse param..";
    vector<string> result;
    std::size_t pos = aParam.find(",");
    if(pos == std::string::npos)
    {
        cout << "WARN: can't parse "<<aParam<<" - just 1 cam or not seperat by ,"<<endl;
        result.push_back(aParam);
    }
    while(pos!=std::string::npos)
    {
        pos = aParam.find(",");
        string temp;
        if (pos!=std::string::npos)
        {
            string temp = aParam.substr(0,pos);
            result.push_back(temp);
            temp = aParam.substr(pos+1,aParam.length());
            aParam = temp;
        }
        else
        {
            result.push_back(aParam);
            break;
        }
    }
    cout<<result.size()<<" params"<<endl;
    return result;
}
//----------------------------------------------------------------------------

int InitOriLinear_main(int argc,char ** argv)
{
    cout<<"************************"<<endl;
    cout<<"*  X : Initial         *"<<endl;
    cout<<"*  X : Orientation     *"<<endl;
    cout<<"*  X : & Position      *"<<endl;
    cout<<"*  X : For Acquisition *"<<endl;
    cout<<"*  X : Linear          *"<<endl;
    cout<<"************************"<<endl;

    string aVecPatNEW, aVecPatREF;
    vector<string> aVecPoseTurn, aPatAngle;
    string aAxeOrient, aOriRef;
    aAxeOrient = "z";
    string aOriOut = "Ori-InitOut";
    bool bWithOriIdentity = false;
    bool forceInPlan = false;
    vector<string> aMulF;
    ElInitArgMain			//initialize Elise, set which is mandantory arg and which is optional arg
    (
    argc,argv,
    //mandatory arguments
    LArgMain()  << EAMC(aOriRef,"Ori folder of reference images",  eSAM_IsExistDirOri)
                << EAMC(aOriOut, "Folder for output initialized orientation- default = Ori-InitOut", eSAM_None)
                << EAMC(aVecPatNEW, "Pattern of new images to orientate PatCam1, PatCam2,..", eSAM_None)
                << EAMC(aVecPatREF, "Pattern of Reference Image = PatRef1, PatRef2,..", eSAM_None),
    //optional arguments
    LArgMain()  <<EAM(aVecPoseTurn, "PatTurn", true, "Images when acquisition have turn [poseTurn1,poseTurn2...]")
                <<EAM(aPatAngle,    "PatAngle", true, "Turn angle [angle1,angle2,...] - + => turn left, - => turn right")
                <<EAM(aMulF,    "mulF", true, "Multiplication factor for adjustment each turn [mul1,mul2,...]")
                 <<EAM(aAxeOrient,    "Axe", true, "Which axe to calcul rotation about - default = z")
                <<EAM(bWithOriIdentity,    "WithIdent", true, "Initialize with orientation identique (default = false)")
                <<EAM(forceInPlan,   "Plan", true, "Force using vector [0,0,1] to initialize (garantie all poses will be in a same plan) - (default = false)")
    );
    if (MMVisualMode) return EXIT_SUCCESS;

    vector<string>aVecPatNEWImg = parseParam(aVecPatNEW);
    vector<string>aVecPatREFImg = parseParam(aVecPatREF);
    vector<double>aVecAngleTurn = parse_dParam(aPatAngle);
    vector<double>MulF;
    if (aMulF.size() > 0)
        MulF = parse_dParam(aMulF);
    if (MulF.size() != aPatAngle.size())
    {
        for (uint i=0; i<(aPatAngle.size() - aMulF.size()); i++)
            MulF.push_back(1.0);
    }

    bool isNumParamOK = checkNumParam(aVecPatNEWImg, aVecPatREFImg, aVecPoseTurn , aVecAngleTurn, MulF);
    vector<SerieCamLinear*> aSystem;

    //creat serie cam
    if (isNumParamOK)
    {
        bool Exist = ELISE_fp::IsDirectory(aOriOut);
        if (!Exist)
        {
            cout<<"Dir "<<aOriOut<<" not exist! Create..";
            ELISE_fp::MkDir(aOriOut);
            cout<<"done"<<endl;
        }
        for (uint i=0; i<aVecPatREFImg.size(); i++)
        {
            string aPatImgRef = aVecPatREFImg[i];
            string aPatImgNew = aVecPatNEWImg[i];
            SerieCamLinear * aCam = new SerieCamLinear(aPatImgRef , aPatImgNew, aOriRef, aOriOut, aAxeOrient, MulF, i);
            aSystem.push_back(aCam);
        }
    }
    else
        return EXIT_SUCCESS;

    //if init without turn
    cout<<"System has "<<aSystem.size()<<" cams"<<endl;
    if (isNumParamOK && (aVecPoseTurn.size() == 0) && (aVecAngleTurn.size() == 0) )
    {
        SerieCamLinear * cam0 = aSystem[0];
        cam0->saveSystem(aSystem);
        cam0->calPosRlt();
        for (uint i=0; i<aSystem.size(); i++)
        {
            SerieCamLinear * cam = aSystem[i];
            cam->calVecMouvement();
        }
        Pt3dr vecMouvCam0 = cam0->mVecMouvement;
        cam0->initSerie(vecMouvCam0 , aVecPoseTurn, aVecAngleTurn);
        for (uint i=1; i<aSystem.size(); i++)
        {
             SerieCamLinear * cam = aSystem[i];
             cam->initSerieByRefSerie(cam0);
        }
    }
    else
    {
    //init with turn
        SerieCamLinear * cam0 = aSystem[0];
        aVecPoseTurn.push_back(cam0->mSetImgNEW.back());    //tricher pour initializer dernier section
        cam0->saveSystem(aSystem);
        cam0->calPosRlt();
        cam0->partageSection(aVecPoseTurn, aVecAngleTurn);
        Pt3dr vecMouv0 = cam0->calVecMouvement();
        cam0->initSerieWithTurn(vecMouv0, aVecPoseTurn, aVecAngleTurn);
        for (uint i=1; i<aSystem.size(); i++)
        {
             SerieCamLinear * cam = aSystem[i];
             cam->initSerieByRefSerie(cam0);
        }
    }

/*
    //separate pattern camera
    std::size_t pos = aVecPatternNewImages_E.find(",");
    std::size_t pos1 = aVecPatternRefImages_E.find(",");
    if(pos == std::string::npos && pos1 == std::string::npos)
    {
        cout << "Warning : Can't seperate Patterns Cameras, maybe system have just 1 camera or user not seperate by ',' sign (Pat_Cam1,Pat_Cam 2,...)"<<endl;
        aVecPatternNewImages.push_back(aVecPatternNewImages_E);
        aVecPatternRefImages.push_back(aVecPatternRefImages_E);
    }
    while(pos!=std::string::npos)
    {
        pos = aVecPatternNewImages_E.find(",");
        pos1 = aVecPatternRefImages_E.find(",");
        string temp, temp1;
        if (pos!=std::string::npos)
        {
            string temp = aVecPatternNewImages_E.substr(0,pos);
            string temp1 = aVecPatternRefImages_E.substr(0,pos1);
            aVecPatternNewImages.push_back(temp);
            aVecPatternRefImages.push_back(temp1);
            temp = aVecPatternNewImages_E.substr(pos+1,aVecPatternNewImages_E.length());
            temp1 = aVecPatternRefImages_E.substr(pos1+1,aVecPatternRefImages_E.length());
            aVecPatternNewImages_E = temp;
            aVecPatternRefImages_E = temp1;
        }
        else
        {
            aVecPatternNewImages.push_back(aVecPatternNewImages_E);
            aVecPatternRefImages.push_back(aVecPatternRefImages_E);
            break;
        }
    }

    for (uint i=0; i<aVecPatternNewImages.size(); i++)
    {
        cout<<aVecPatternNewImages[i]<< " ++ " << aVecPatternRefImages[i]<<endl;
    }

    //separate changing direction acquisition
    vector<string> aVecPoseTurn;
    vector<string> aVecAngle;
    pos = aPatPoseTurn.find(",");
    pos1 = aPatAngle.find(",");
    if(pos == std::string::npos && pos1 == std::string::npos && aPatPoseTurn.length() > 0)
    {
        cout << "Warning : Can't seperate Patterns of pose turn and angle turn, maybe system have just 1 turn"<<endl;
        aVecPoseTurn.push_back(aPatPoseTurn);
        aVecAngle.push_back(aPatAngle);
    }
    while(pos!=std::string::npos)
    {
        pos = aPatPoseTurn.find(",");
        pos1 = aPatAngle.find(",");
        string temp, temp1;
        if (pos!=std::string::npos)
        {
            string temp = aPatPoseTurn.substr(0,pos);
            string temp1 = aPatAngle.substr(0,pos1);
            aVecPoseTurn.push_back(temp);
            aVecAngle.push_back(temp1);
            temp = aPatPoseTurn.substr(pos+1,aPatPoseTurn.length());
            temp1 = aPatAngle.substr(pos1+1,aVecPatternRefImages_E.length());
            aPatPoseTurn = temp;
            aPatAngle = temp1;
        }
        else
        {
            aVecPoseTurn.push_back(aPatPoseTurn);
            aVecAngle.push_back(aPatAngle);
            break;
        }
    }
    vector<double> aVecAngle_Dec;
    for (uint i=0; i<aVecPoseTurn.size(); i++)
    {
        double digit = ToDecimal(aVecAngle[i]);
        aVecAngle_Dec.push_back(digit);
        cout<<aVecPoseTurn[i]<< " ++ " << digit<<endl;
    }

    //==== Test new code Giang 21/3/2016 ====//
    cout<<"System with "<<aVecPatternNewImages.size()<<" cameras"<<endl;
    if (aVecPatternNewImages.size() >= 1)
    {
//        double xOffsetRef = 0;double yOffsetRef = 0;double zOffsetRef = 0;
//        double xOffset = 0;double yOffset = 0;double zOffset = 0;
        std::vector<cOrientationConique> aRefOriList;
        std::vector<cOrientationConique> aOriConique1stCam;

        for(uint ii=0; ii<aVecPatternNewImages.size(); ii++)
        {   //lire chaque pattern d'image de chaque camera
            cout<<"\nInit Cam "<<ii<<" : ";
            aFullPatternNewImages = RequireFromString<string>(aVecPatternNewImages[ii],"Pat Cam");
            aFullPatternRefImages = RequireFromString<string>(aVecPatternRefImages[ii],"Pat Cam");
            cout<<"***"<<aFullPatternNewImages<<"***"<<endl;

            // Initialize name manipulator & files
            std::string aDirNewImages,aDirRefImages, aPatNewImages,aPatRefImages;
            SplitDirAndFile(aDirNewImages,aPatNewImages,aFullPatternNewImages);
            SplitDirAndFile(aDirRefImages,aPatRefImages,aFullPatternRefImages);
            StdCorrecNameOrient(aOriRef,aDirRefImages);//remove "Ori-" if needed

            cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirNewImages);
            std::vector<std::string> aSetNewImages = *(aICNM->Get(aPatNewImages));

            std::cout<<"\nInit images:\n";
            //nouvelle image a initializer
            for (unsigned int ik=0;ik<aSetNewImages.size();ik++)
                std::cout<<"  - "<<aSetNewImages[ik]<<"\n";

            aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirRefImages);
            const std::vector<std::string> aSetRefImages = *(aICNM->Get(aPatRefImages));

            std::cout<<"\nRef images ("<<aPatRefImages<<"):\n";
            for (unsigned int k=0;k<aSetRefImages.size();k++)
                std::cout<<" - "<<aSetRefImages[k]<<"\n";
            
            ELISE_ASSERT(aSetRefImages.size()>1,"Number of reference image must be > 1");   //pour chaque camera, il fault au moins 2 images pour caculer vector de deplacement

            //Read orientation initial (first image in series)
            string aOriRefImage="Ori-"+aOriRef+"/Orientation-"+aSetRefImages.back()+".xml";
            cOrientationConique aOriConiqueRef=StdGetFromPCP(aOriRefImage,OrientationConique);
            cOrientationConique aOriConique;
            //init relative position b/w different series image
            Pt3dr VecMouvement;
            VecMouvement = CalVecAvancementInit(aSetRefImages,aOriRef);
            //====Test===//
            if (forceInPlan)
                {VecMouvement = Pt3dr(0,0,1);}
            //===========//
            vector<Section> aSection;
            if (ii==0) //1st camera as reference
            {
                std::string aOriRefImage="Ori-"+aOriRef+"/Orientation-"+aSetRefImages.back()+".xml";
                aOriConique = StdGetFromPCP(aOriRefImage,OrientationConique); //prendre orientation Conique partie a partir de XML fichier
                aRefOriList.push_back(aOriConique);
                std::cout<<aOriConique.Externe().Centre()<<"\n";
                if (aVecPoseTurn.size() > 0)
                {
                    //creat section for turn initialization
                    for (uint f=0; f<=aVecPoseTurn.size(); f++)
                    {
                        Section thisSection;
                        string PoseInit;
                        string PoseEnd;
                        double indPoseInit, indPoseEnd;
                        if (f == 0)
                        {
                            PoseInit = aSetNewImages.front();
                            indPoseInit = 0;
                            std::vector<string>::iterator it;
                            it = std::find(aSetNewImages.begin(), aSetNewImages.end(), aVecPoseTurn[f]);
                            double p = std::distance(aSetNewImages.begin(), it );
                            //bool isPresent = (it != aSetNewImages.end());
                            PoseEnd = aSetNewImages[p-1];
                            indPoseEnd = p-1;
                            //cout<<"Section "<<f<<": "<<PoseInit<<" - "<<PoseEnd<<" turn "<<"0 (section reference)"<<endl;
                            for(uint j=indPoseInit;j<=indPoseEnd;j++)
                            {thisSection.Poses.push_back(aSetNewImages[j]);}
                            thisSection.angle = 0;
                            thisSection.isReference=true;
                            aSection.push_back(thisSection);
                        }
                        else
                        {
                            PoseInit = aVecPoseTurn[f-1];
                            std::vector<string>::iterator it;
                            if (f < aVecPoseTurn.size())
                            {
                                it = std::find(aSetNewImages.begin(), aSetNewImages.end(), aVecPoseTurn[f]);
                                double p = std::distance(aSetNewImages.begin(), it );
                                PoseEnd = aSetNewImages[p-1];
                                indPoseEnd=p-1;
                                it = std::find(aSetNewImages.begin(), aSetNewImages.end(), PoseInit);
                                p = std::distance(aSetNewImages.begin(), it );
                                indPoseInit = p;
                                for(uint j=indPoseInit;j<=indPoseEnd;j++)
                                {thisSection.Poses.push_back(aSetNewImages[j]);}
                                thisSection.angle = aVecAngle_Dec[f-1];
                                thisSection.isReference=false;
                                aSection.push_back(thisSection);
                            }
                            else
                            {
                                it = std::find(aSetNewImages.begin(), aSetNewImages.end(), PoseInit);
                                double p = std::distance(aSetNewImages.begin(), it );
                                indPoseInit = p;
                                PoseEnd = aSetNewImages.back();
                                for(uint j=indPoseInit;j<aSetNewImages.size();j++)
                                {thisSection.Poses.push_back(aSetNewImages[j]);}
                                thisSection.angle = aVecAngle_Dec[f-1];
                                thisSection.isReference=false;
                                aSection.push_back(thisSection);
                            }
                            //cout<<"Section "<<f<<": "<<PoseInit<<" - "<<PoseEnd<<" turn "<<aVecAngle_Dec[f-1]<<endl;
                        }
                    }
                }
                else
                {//no turn in initialization
                    Section thisSection;
                    thisSection.angle = 0;
                    thisSection.isReference = true;
                    thisSection.Poses = aSetNewImages;
                    aSection.push_back(thisSection);
                }
                for (uint j=0; j<aSection.size(); j++)
                {
                    cout<<endl<<"Section "<<j<<": "<<aSection[j].Poses.front()<<" - "<<aSection[j].Poses.back()<<" - "<<aSection[j].angle<< " degree"<<endl;
                    if (aSection[j].isReference)
                    {
                        OrientationLinear(aSection[j].Poses, VecMouvement, aOriConique, aOriOut);
                    }
                    else
                    {
                        //take last pose of section that just initialized as XML reference
                        std::string aOriRefSection="Ori-"+aOriOut+"/Orientation-"+aSection[j-1].Poses.back()+".xml";
                        cOrientationConique aXMLRef = StdGetFromPCP(aOriRefSection,OrientationConique);
                        //aXMLRef.Externe().ParamRotation().CodageMatr().SetVal(aOriRefSection.Externe().ParamRotation().CodageMatr().Val());
                        //calculate orientation turn of new section
                        matrix orientationRef;
                        matrix orientationSection;
                        set_matrixLine(orientationRef, aXMLRef.Externe().ParamRotation().CodageMatr().Val().L1(), 0);
                        set_matrixLine(orientationRef, aXMLRef.Externe().ParamRotation().CodageMatr().Val().L2(), 1);
                        set_matrixLine(orientationRef, aXMLRef.Externe().ParamRotation().CodageMatr().Val().L3(), 2);
                        cout<<"Avant:\n    "<<aXMLRef.Externe().ParamRotation().CodageMatr().Val().L1()<<"\n    "
                            <<aXMLRef.Externe().ParamRotation().CodageMatr().Val().L2()<<"\n    "
                            <<aXMLRef.Externe().ParamRotation().CodageMatr().Val().L3()<<endl;
                        CalOrient(orientationRef, aSection[j].angle*PI/180, orientationSection, aAxeOrient);
                        //RotationParAxe(VecMouvement,0, orientationSection);
                        aXMLRef.Externe().ParamRotation().CodageMatr().Val().L1() = get_matrixLine(orientationSection,0);
                        aXMLRef.Externe().ParamRotation().CodageMatr().Val().L2() = get_matrixLine(orientationSection,1);
                        aXMLRef.Externe().ParamRotation().CodageMatr().Val().L3() = get_matrixLine(orientationSection,2);
                        cout<<"Apres:\n    "<<aXMLRef.Externe().ParamRotation().CodageMatr().Val().L1()<<"\n    "
                            <<aXMLRef.Externe().ParamRotation().CodageMatr().Val().L2()<<"\n    "
                            <<aXMLRef.Externe().ParamRotation().CodageMatr().Val().L3()<<endl;
                        //calculate new vector mouvement of section
                        cout<<"Vec mouv avant = "<<VecMouvement;
                        //VecMouvement = mult_vector(orientationSection, VecMouvement);
                        VecMouvement = CalDirectionVecMouvement(VecMouvement, aSection[j].angle*PI/180, aAxeOrient);
                        cout<<" - apres = "<<VecMouvement<<endl;
                        aXMLRef.Externe().Centre() = aXMLRef.Externe().Centre() + VecMouvement;
                        //initialize Linear new section
                        OrientationLinear(aSection[j].Poses, VecMouvement, aXMLRef, aOriOut);
                    }
                }
            }

        }
    }



    //===========Old Code stable=================//
    cout<<"System with "<<aVecPatternNewImages.size()<<" cameras"<<endl;
    if (aVecPatternNewImages.size() >= 1)
    {
        double xOffsetRef = 0;double yOffsetRef = 0;double zOffsetRef = 0;
        double xOffset = 0;double yOffset = 0;double zOffset = 0;
        std::vector<cOrientationConique> aRefOriList;
        std::vector<cOrientationConique> aOriConique1stCam;
    for(uint ii=0; ii<aVecPatternNewImages.size(); ii++)
    {   //lire chaque pattern d'image de chaque camera
        cout<<"\nInit Cam "<<ii<<" : ";
        aFullPatternNewImages = RequireFromString<string>(aVecPatternNewImages[ii],"Pat Cam");
        aFullPatternRefImages = RequireFromString<string>(aVecPatternRefImages[ii],"Pat Cam");
        cout<<"***"<<aFullPatternNewImages<<"***"<<endl;

        // Initialize name manipulator & files
        std::string aDirNewImages,aDirRefImages, aPatNewImages,aPatRefImages;
        SplitDirAndFile(aDirNewImages,aPatNewImages,aFullPatternNewImages);
        SplitDirAndFile(aDirRefImages,aPatRefImages,aFullPatternRefImages);
        StdCorrecNameOrient(aOriRef,aDirRefImages);//remove "Ori-" if needed

        cInterfChantierNameManipulateur * aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirNewImages);
        const std::vector<std::string> aSetNewImages = *(aICNM->Get(aPatNewImages));

        std::cout<<"\nInit images:\n";
        //nouvelle image a initializer
        for (unsigned int ik=0;ik<aSetNewImages.size();ik++)
            std::cout<<"  - "<<aSetNewImages[ik]<<"\n";


        aICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirRefImages);
        const std::vector<std::string> aSetRefImages = *(aICNM->Get(aPatRefImages));

        ELISE_ASSERT(aSetRefImages.size()>1,"Number of reference image must be > 1");   //pour chaque camera, il fault au moins 2 images pour caculer vector de deplacement

        std::cout<<"\nRef images:\n";
        //Read orientation initial (first image in series)
        string aOriRefImage="Ori-"+aOriRef+"/Orientation-"+aSetRefImages.back()+".xml";
        cOrientationConique aOriConiqueRef=StdGetFromPCP(aOriRefImage,OrientationConique);
        //init relative position b/w different series image
        if (ii==0) //1st camera as reference
        {
            xOffset = yOffset = zOffset = 0;
            xOffsetRef = aOriConiqueRef.Externe().Centre().x;
            yOffsetRef = aOriConiqueRef.Externe().Centre().y;
            zOffsetRef = aOriConiqueRef.Externe().Centre().z;
            double xBefore=0, yBefore=0, zBefore=0;
            double xAcc = 0, yAcc = 0, zAcc = 0;
            for (unsigned int i=0;i<aSetRefImages.size();i++)
            {   //tout les poses references dans camera
                std::cout<<"  - "<<aSetRefImages[i]<<" ";
                std::string aOriRefImage="Ori-"+aOriRef+"/Orientation-"+aSetRefImages[i]+".xml";
                cOrientationConique aOriConique=StdGetFromPCP(aOriRefImage,OrientationConique); //prendre orientation Conique partie a partir de XML fichier
                aRefOriList.push_back(aOriConique);
                std::cout<<aOriConique.Externe().Centre()<<"\n";
                if (i==0)
                {   //1st pose as reference
                    xBefore = aOriConique.Externe().Centre().x;
                    yBefore = aOriConique.Externe().Centre().y;
                    zBefore = aOriConique.Externe().Centre().z;
                }
                xAcc = xAcc + aOriConique.Externe().Centre().x - xBefore;
                yAcc = yAcc + aOriConique.Externe().Centre().y - yBefore;
                zAcc = zAcc + aOriConique.Externe().Centre().z - zBefore;
                xBefore =  aOriConique.Externe().Centre().x;
                yBefore = aOriConique.Externe().Centre().y;
                zBefore = aOriConique.Externe().Centre().z;
            }
            //compute orientation and movement
            double xMov = xAcc/(aSetRefImages.size()-1);
            double yMov = yAcc/(aSetRefImages.size()-1);
            double zMov = zAcc/(aSetRefImages.size()-1);
            cout<<endl<<"Init with vector movement = "<<xMov<<" ; "<<yMov<<" ; "<<zMov<<" ; "<<endl;
            //Create a XML file with class cOrientationConique (define in ParamChantierPhotogram.h)
            double xEstimate = aRefOriList.front().Externe().Centre().x;
            double yEstimate = aRefOriList.front().Externe().Centre().y;
            double zEstimate = aRefOriList.front().Externe().Centre().z;
            cOrientationConique aOriConique = aRefOriList.front();
            //std::cout<<"\nInit Images:\n";
            for (unsigned int i=0;i<aSetNewImages.size();i++)
            {
                    aOriConique.Externe().Centre().x = xEstimate;
                    aOriConique.Externe().Centre().y = yEstimate;
                    aOriConique.Externe().Centre().z = zEstimate;
                    xEstimate = xEstimate + xMov;
                    yEstimate = yEstimate + yMov;
                    zEstimate = zEstimate + zMov;

                aOriConique.Externe().ParamRotation().CodageMatr().SetVal(aOriConiqueRef.Externe().ParamRotation().CodageMatr().Val());
                MakeFileXML(aOriConique, "Ori-"+aOriOut+"/Orientation-"+aSetNewImages[i]+".xml");
                aOriConique1stCam.push_back(aOriConique);
            }
        }

//        for(uint k=0; k<aVecPoseTurn.size(); k++)
//        {
//            if (aVecPoseTurn[i] == aSetNewImages[i])
//            {
//                matrix orientationRef; matrix orientationTurn;
//                set_matrixLine(orientationRef, aOriConiqueRef.Externe().ParamRotation().CodageMatr().Val().L1(), 0);
//                set_matrixLine(orientationRef, aOriConiqueRef.Externe().ParamRotation().CodageMatr().Val().L2(), 1);
//                set_matrixLine(orientationRef, aOriConiqueRef.Externe().ParamRotation().CodageMatr().Val().L3(), 2);
//                CalOrient(orientationRef, aVecAngle_Dec[i] , orientationTurn);
//                aOriConiqueRef.Externe().ParamRotation().CodageMatr().Val().L1() = get_matrixLine(orientationTurn, 1);
//                aOriConiqueRef.Externe().ParamRotation().CodageMatr().Val().L2() = get_matrixLine(orientationTurn, 2);
//                aOriConiqueRef.Externe().ParamRotation().CodageMatr().Val().L3() = get_matrixLine(orientationTurn, 3);
//            }
//        }

        else    //others camera in series, initialize by offset with cam 1
        {
                string aOriRefImage = "Ori-"+aOriRef+"/Orientation-"+aSetRefImages.front()+".xml";
                cOrientationConique aOriConiqueThisCam = StdGetFromPCP(aOriRefImage,OrientationConique);
                //Read orientation initial (first image in series of cam 1)
                cOrientationConique aOriConiqueRefCam1 = aOriConique1stCam.front();
                //offset b/w Cam 1 and this camera
                xOffset = aOriConiqueThisCam.Externe().Centre().x - aOriConiqueRefCam1.Externe().Centre().x;
                yOffset = aOriConiqueThisCam.Externe().Centre().y - aOriConiqueRefCam1.Externe().Centre().y;
                zOffset = aOriConiqueThisCam.Externe().Centre().z - aOriConiqueRefCam1.Externe().Centre().z;
                cout<<"Offset = "<<xOffset<<" - "<<yOffset<<" - "<<zOffset<<endl;
                for (unsigned int i=0;i<aSetNewImages.size();i++)
                {
                    cOrientationConique  aOriConique  = aOriConique1stCam[i];
                    aOriConique.Externe().Centre().x = aOriConique.Externe().Centre().x + xOffset;
                    aOriConique.Externe().Centre().y = aOriConique.Externe().Centre().y + yOffset;
                    aOriConique.Externe().Centre().z = aOriConique.Externe().Centre().z + zOffset;
                    aOriConique.Externe().ParamRotation().CodageMatr().SetVal(aOriConiqueThisCam.Externe().ParamRotation().CodageMatr().Val());
                    MakeFileXML(aOriConique, "Ori-"+aOriOut+"/Orientation-"+aSetNewImages[i]+".xml");
                }
        }
    }
    }
    */
    return EXIT_SUCCESS;

}

/* Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant Ã  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est rÃ©gi par la licence CeCILL-B soumise au droit franÃ§ais et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusÃ©e par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilitÃ© au code source et des droits de copie,
de modification et de redistribution accordÃ©s par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitÃ©e.  Pour les mÃªmes raisons,
seule une responsabilitÃ© restreinte pÃ¨se sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concÃ©dants successifs.

A cet Ã©gard  l'attention de l'utilisateur est attirÃ©e sur les risques
associÃ©s au chargement,  Ã  l'utilisation,  Ã  la modification et/ou au
dÃ©veloppement et Ã  la reproduction du logiciel par l'utilisateur Ã©tant
donnÃ© sa spÃ©cificitÃ© de logiciel libre, qui peut le rendre complexe Ã
manipuler et qui le rÃ©serve donc Ã  des dÃ©veloppeurs et des professionnels
avertis possÃ©dant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invitÃ©s Ã  charger  et  tester  l'adÃ©quation  du
logiciel Ã  leurs besoins dans des conditions permettant d'assurer la
sÃ©curitÃ© de leurs systÃ¨mes et ou de leurs donnÃ©es et, plus gÃ©nÃ©ralement,
Ã  l'utiliser et l'exploiter dans les mÃªmes conditions de sÃ©curitÃ©.

Le fait que vous puissiez accÃ©der Ã  cet en-tÃªte signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez acceptÃ© les
termes.
Footer-MicMac-eLiSe-25/06/2007/*/
