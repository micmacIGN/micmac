#include "TaskCorrel.h"
#include <stdio.h>


    /******************************************************************************
    The main function.
    ******************************************************************************/
int TaskCorrel_main(int argc,char ** argv)
{
    cout<<"********************************************************"<<endl;
    cout<<"*    TaskCorrel - creat XML for TiepTri                *"<<endl;
    cout<<"********************************************************"<<endl;
        string pathPlyFileS ;
        string aFullPattern, aOriInput;
        bool assum1er=false;
        bool useExistHomoStruct = false;
        double aAngleF = 90;
        string aDirXML = "XML_TiepTri";
        string xmlCpl = "PairHomol.xml";
        bool Test=false;
        int nInteraction = 0;
        double aZ = 0.25;
        double aSclElps = -1.0;
        double distMax = TT_DISTMAX_NOLIMIT;
        int rech = TT_DEF_SCALE_ZBUF;
        Pt3dr clIni(255.0,255.0,255.0);
        ElInitArgMain
                (
                    argc,argv,
                    //mandatory arguments
                    LArgMain()
                    << EAMC(aFullPattern, "Pattern of images",  eSAM_IsPatFile)
                    << EAMC(aOriInput, "Input Initial Orientation",  eSAM_IsExistDirOri)
                    << EAMC(pathPlyFileS, "path to mesh(.ply) file - created by Inital Ori", eSAM_IsExistFile),
                    //optional arguments
                    LArgMain()
                    << EAM(xmlCpl, "xmlCpl", true, "file contain couple of image - processe by couple")
                    << EAM(assum1er, "assum1er", true, "always use 1er pose as img master, default=0")
                    << EAM(useExistHomoStruct, "useExist", true, "use exist homol struct - default = false")
                    << EAM(aAngleF, "angleV", true, "limit view angle - default = 90 (all triangle is viewable)")
                    << EAM(aDirXML, "OutXML", true, "Output directory for XML File. Default = XML_TiepTri")
                    << EAM(Test, "Test", true, "Test stretching")
                    << EAM(nInteraction, "nInt", true, "nInteraction")
                    << EAM(aZ, "aZ", true, "aZoom image display")
                    << EAM(aSclElps, "aZEl", true, "fix size ellipse display (in pxl)")
                    << EAM(clIni, "clIni", true, "color mesh (=[255,255,255])")
                    << EAM(distMax, "distMax", true, "Limit distant process from camera")
                    << EAM(rech, "rech", true, "calcul ZBuffer in Reechantilonage (def=2)")
                    );

        if (MMVisualMode) return EXIT_SUCCESS;

        std::string aDir,aNameImg;
        SplitDirAndFile(aDir,aNameImg,aFullPattern);
        StdCorrecNameOrient(aOriInput,aDir);
        cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

        if (EAMIsInit(& xmlCpl))
        {
            //creat xml file couple by couple
            cAppliTaskCorrelByXML * aAppli = new cAppliTaskCorrelByXML(xmlCpl, aICNM, aDir, aOriInput, aNameImg, pathPlyFileS);
            aAppli->DoAllCpl();
            aAppli->ExportXML(aDirXML);
        }
        else
        {
            cAppliTaskCorrel * aAppli = new cAppliTaskCorrel(aICNM , aDir, aOriInput, aNameImg);
            aAppli->lireMesh(pathPlyFileS, aAppli->VTri(), aAppli->VTriF());
            aAppli->SetNInter(nInteraction, aZ);
            aAppli->Rech() = 1.0/double (rech);
            aAppli->DistMax() = distMax;
            aAppli->NameMesh() = pathPlyFileS;
            aAppli->ZBuffer();
            aAppli->DoAllTri();
            aAppli->ExportXML(aDirXML, clIni);
        }

        return EXIT_SUCCESS;
    }
