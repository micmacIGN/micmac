#include "TaskCorrel.h"
#include "GCPByMesh.h"
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
        string aMesureXML = "";
        string aDirXML = "XML_TiepTri";
        string xmlCpl = "PairHomol.xml";
        bool Test=false;
        int nInteraction = 0;
        double aZ = 0.25;
        double aSclElps = -1.0;
        double distMax = TT_DISTMAX_NOLIMIT;
        int rech = TT_DEF_SCALE_ZBUF;
        Pt3dr clIni(255.0,255.0,255.0);
        bool noTif = false;
        bool keepAll2nd = false;
        double MD_SEUIL_SURF_TRIANGLE = TT_SEUIL_SURF_TRIANGLE;
        int MethodZBuf = 3;
        bool Safe = true;
        bool InverseOrder = false;
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
                    << EAM(MethodZBuf, "MethodZBuf", true, "method of grab pixel in triangle (1=very good (low), 3=fast (quite good - def))")
                    << EAM(Safe, "Safe", true, "Safe (but longer) in ZBuffer calcul by test visible in image (def=true)")
                    << EAM(keepAll2nd, "keepAll2nd", true, "Don't filter image 2nd")
                    << EAM(MD_SEUIL_SURF_TRIANGLE, "surfTri", true, "Threshold of surface to filter triangle too small (def=100)")
                    << EAM(InverseOrder, "InvOrder", true, "Inverse order of triangle's vertices (as we don't know how 's the mesh generated - def=false")
                    );

        if (MMVisualMode) return EXIT_SUCCESS;

        std::string aDir,aNameImg;
        SplitDirAndFile(aDir,aNameImg,aFullPattern);
        StdCorrecNameOrient(aOriInput,aDir);

        cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

        //===========Modifier ou chercher l'image si l'image ne sont pas tif============//
           std::size_t found = aFullPattern.find_last_of(".");
           string ext = aFullPattern.substr(found+1);
           cout<<"Ext : "<<ext<<endl;
           if ( ext.compare("tif") )   //ext equal tif
           {
               noTif = true;
               cout<<" No Tif"<<endl;
           }
           if (noTif)
           {
               vector<string> mVName = *(aICNM->Get(aNameImg));
               list<string> cmd;
               for (uint aK=0; aK<mVName.size(); aK++)
               {
                    string aCmd = MM3DStr +  " PastDevlop "+  mVName[aK] + " Sz1=-1 Sz2=-1 Coul8B=0";
                    cmd.push_back(aCmd);
               }
               cEl_GPAO::DoComInParal(cmd);
               mVName.clear();
           }
        //===============================================================================/

        if (EAMIsInit(& xmlCpl))
        {
            //creat xml file couple by couple
            cAppliTaskCorrelByXML * aAppli = new cAppliTaskCorrelByXML(xmlCpl, aICNM, aDir, aOriInput, aNameImg, pathPlyFileS, noTif);
            aAppli->DoAllCpl();
            aAppli->ExportXML(aDirXML);
        }
        else
        {
            /*
            cParamAppliTaskCorrel * aParam = new cParamAppliTaskCorrel(
                                                                       aICNM,
                                                                       aDir,
                                                                       aOriInput,
                                                                       aFullPattern,
                                                                       noTif,
                                                                       aMesureXML
                                                                      );
                                                                      */
            cAppliTaskCorrel * aAppli = new cAppliTaskCorrel(aICNM , aDir, aOriInput, aNameImg, noTif);

            aAppli->SafeZBuf() = Safe;
            aAppli->ZBuf_InverseOrder() = InverseOrder;
            aAppli->KeepAll2nd()=keepAll2nd;
            aAppli->SEUIL_SURF_TRIANGLE()=MD_SEUIL_SURF_TRIANGLE;
            aAppli->MethodZBuf()=MethodZBuf;
            aAppli->lireMesh(pathPlyFileS/*, aAppli->VTri(), aAppli->VTriF()*/);
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

/* ***===***===***===***===***===***===***===***===***===***===***===***===***===***===***===
 *          Task Correl receive pts interet also - Saisie points appuis manier Tieptri
 * ***===***===***===***===***===***===***===***===***===***===***===***===***===***===***=== */


int TaskCorrelWithPts_main(int argc,char ** argv)
{
    cout<<"****************************************************************************"<<endl;
    cout<<"*    TaskCorrelWithPts - creat XML for TiepTri with Pts Interest predefini *"<<endl;
    cout<<"****************************************************************************"<<endl;
        string pathPlyFileS ;
        string aFullPattern, aOriInput;

        string aDirXML = "XML_TiepTri_GCP";
        string aMesureXML = "Mesured_GCPTiepTri.xml";

        bool noTif = false;

        double distMax = TT_DISTMAX_NOLIMIT;
        int rech = TT_DEF_SCALE_ZBUF;
        int MethodZBuf = 3;
        double MD_SEUIL_SURF_TRIANGLE = TT_SEUIL_SURF_TRIANGLE;

        ElInitArgMain
                (
                    argc,argv,
                    //mandatory arguments
                    LArgMain()
                    << EAMC(aFullPattern, "Pattern of images",  eSAM_IsPatFile)
                    << EAMC(aOriInput, "Input Initial Orientation",  eSAM_IsExistDirOri)
                    << EAMC(pathPlyFileS, "path to mesh(.ply) file - created by Inital Ori", eSAM_IsExistFile)
                    << EAM(aMesureXML, "GCP", true, "mesure image file xml", eSAM_IsExistFile),
                    //optional arguments
                    LArgMain()
                    << EAM(aDirXML, "OutXML", true, "Output directory for XML File. Default = XML_TiepTri_GCP")
                    << EAM(distMax, "distMax", true, "Limit distant process from camera")
                    << EAM(rech, "rech", true, "calcul ZBuffer in Reechantilonage (def=2)")
                    << EAM(MethodZBuf, "MethodZBuf", true, "method of grab pixel in triangle (1=very good (low), 3=fast (quite good - def))")
                    << EAM(MD_SEUIL_SURF_TRIANGLE, "surfTri", true, "Threshold of surface to filter triangle too small (def=100)")
                    );

        if (MMVisualMode) return EXIT_SUCCESS;

        std::string aDir,aNameImg;
        SplitDirAndFile(aDir,aNameImg,aFullPattern);
        StdCorrecNameOrient(aOriInput,aDir);

        cInterfChantierNameManipulateur * aICNM = cInterfChantierNameManipulateur::BasicAlloc(aDir);

        //======Read XML GCP file, sort by images=======
        vector<cOneImgMesure*> aVImgMesure;
        ReadXMLMesurePts(aMesureXML, aVImgMesure);

        //===========Modifier ou chercher l'image si l'image ne sont pas tif============//
           std::size_t found = aFullPattern.find_last_of(".");
           string ext = aFullPattern.substr(found+1);
           cout<<"Ext : "<<ext<<endl;
           if ( ext.compare("tif") )   //ext equal tif
           {
               noTif = true;
               cout<<" No Tif"<<endl;
           }
           if (noTif)
           {
               vector<string> mVName = *(aICNM->Get(aNameImg));
               list<string> cmd;
               for (uint aK=0; aK<mVName.size(); aK++)
               {
                    string aCmd = MM3DStr +  " PastDevlop "+  mVName[aK] + " Sz1=-1 Sz2=-1 Coul8B=0";
                    cmd.push_back(aCmd);
               }
               cEl_GPAO::DoComInParal(cmd);
               mVName.clear();
           }
        //===============================================================================/

        return EXIT_SUCCESS;
    }
