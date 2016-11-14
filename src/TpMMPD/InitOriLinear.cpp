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
#include "StdAfx.h"
double ToDecimal(string aString)
{
    uint mLength = aString.length();
    vector<int> aVecDigit;
    bool negativ = false;
    double sum = 0;
    for (uint i=0; i<aString.length(); i++)
    {
        if (aString[i] != '-')
        {
            int digit = aString[i]-'0';
            aVecDigit.push_back(digit);
            //cout << digit <<endl;
        }
        else
        {
            negativ=true;
        }
    }

    mLength=aVecDigit.size();
    for (uint i=0; i<mLength; i++)
    {
        int h = aVecDigit.back();
        sum = sum + h * pow(10.,(double)i);
        aVecDigit.pop_back();
    }
    if (negativ)
        { sum = 0-sum;}
    return sum;
}

typedef double matrix[3][3];
void set_matrixLine(matrix mat_1 , Pt3dr line, int indx)
{
    if (indx == 0)
    {
        mat_1[0][0] = line.x ; mat_1[0][1] = line.y ; mat_1[0][2] = line.z;
    }
    if (indx == 1)
    {
        mat_1[1][0] = line.x ; mat_1[1][1] = line.y ; mat_1[1][2] = line.z;
    }
    if (indx == 2)
    {
        mat_1[2][0] = line.x ; mat_1[2][1] = line.y ; mat_1[2][2] = line.z;
    }
}
void set_matrixLig(matrix mat_1 , Pt3dr lig, int indx)
{
    if (indx == 0)
    {
        mat_1[0][0] = lig.x ; mat_1[1][0] = lig.y ; mat_1[2][0] = lig.z;
    }
    if (indx == 1)
    {
        mat_1[0][1] = lig.x ; mat_1[1][1] = lig.y ; mat_1[2][1] = lig.z;
    }
    if (indx == 2)
    {
        mat_1[0][2] = lig.x ; mat_1[1][2] = lig.y ; mat_1[2][2] = lig.z;
    }
}
Pt3dr get_matrixLine (matrix mat_1, int indx)
{
    Pt3dr Line;
    Line.x = mat_1[indx][0];
    Line.y = mat_1[indx][1];
    Line.z = mat_1[indx][2];
    return Line;
}
void mult_matrix(matrix mat_1, matrix mat_2, matrix fin_mat)
{
   double temp = 0;
   int a, b, c;

   for(a = 0; a < 3; a++)
   {
       for(b = 0; b < 3; b++)
       {
           for(c = 0; c < 3; c++)
           {
               temp += mat_1[b][c] * mat_2[c][a];
           }
           fin_mat[b][a] = temp;
           temp = 0;
       }
   }
}
void mult_matrix(matrix mat_1, matrix mat_2, cTypeCodageMatr result)
{
    matrix fin_mat;
   double temp = 0;
   int a, b, c;

   for(a = 0; a < 3; a++)
   {
       for(b = 0; b < 3; b++)
       {
           for(c = 0; c < 3; c++)
           {
               temp += mat_1[b][c] * mat_2[c][a];
           }
           fin_mat[b][a] = temp;
           temp = 0;
       }
   }
   result.L1()  = get_matrixLine(fin_mat,0);
   result.L2()  = get_matrixLine(fin_mat,1);
   result.L3()  = get_matrixLine(fin_mat,2);
}
Pt3dr mult_vector(matrix mat_1, Pt3dr vec_1)
{
    Pt3dr result;
    result.x = mat_1[0][0]*vec_1.x + mat_1[0][1]*vec_1.y + mat_1[0][2]*vec_1.z;
    result.y = mat_1[1][0]*vec_1.x + mat_1[1][1]*vec_1.y + mat_1[1][2]*vec_1.z;
    result.z = mat_1[2][0]*vec_1.x + mat_1[2][1]*vec_1.y + mat_1[2][2]*vec_1.z;
    return result;
}
void CalOrient(matrix org, double alpha, matrix out, string axe)
{
    cout<<"Matrix rotation axe "<<axe<<" "<<alpha<< " rad"<<endl;
    matrix orient;
    Pt3dr line0, line1, line2;
    if (axe == "x")
    {
        line0 = Pt3dr(1,0 , 0);
        line1 = Pt3dr (0,cos(alpha) ,-sin(alpha) );
        line2 = Pt3dr (0,sin(alpha) ,cos(alpha) );
    }
    if (axe == "y")
    {
        line0 = Pt3dr(cos(alpha),0 , sin(alpha));
        line1 = Pt3dr(0,1,0 );
        line2 = Pt3dr(-sin(alpha),0,cos(alpha) );
    }
    if (axe == "z")
    {
        line0 = Pt3dr(cos(alpha) ,-sin(alpha), 0);
        line1 = Pt3dr(sin(alpha) ,cos(alpha),0 );
        line2 = Pt3dr(0,0,1 );
    }
    set_matrixLine(orient, line0 ,0);
    set_matrixLine(orient, line1 ,1);
    set_matrixLine(orient, line2 ,2);
    cout<<"modif:\n    "<<line0<<"\n    "<<line1<<"\n    "<<line2<<endl;
    mult_matrix(orient,org, out);
}


Pt3dr CalDirectionVecMouvement(Pt3dr VecMouvement, double alpha, string axe)
{
    matrix orient;
    Pt3dr line0, line1, line2;
    if (axe == "x")
    {
        line0 = Pt3dr(1,0 , 0);
        line1 = Pt3dr (0,cos(alpha) ,-sin(alpha) );
        line2 = Pt3dr (0,sin(alpha) ,cos(alpha) );
    }
    if (axe == "y")
    {
        line0 = Pt3dr(cos(alpha),0 , sin(alpha));
        line1 = Pt3dr(0,1,0 );
        line2 = Pt3dr(-sin(alpha),0,cos(alpha) );
    }
    if (axe == "z")
    {
        line0 = Pt3dr(cos(alpha) ,-sin(alpha), 0);
        line1 = Pt3dr(sin(alpha) ,cos(alpha),0 );
        line2 = Pt3dr(0,0,1 );
    }
    set_matrixLine(orient, line0 ,0);
    set_matrixLine(orient, line1 ,1);
    set_matrixLine(orient, line2 ,2);
    return mult_vector(orient,VecMouvement);
}

void OrientationLinear (vector<string> PoseToInit, Pt3dr vectorAvancement, cOrientationConique OriRef, string aOriOut, matrix Ori)
{
    cout<<"Init with vector: "<<vectorAvancement<<endl;
    cOrientationConique aOriConique = OriRef;
    aOriConique.Externe().ParamRotation().CodageMatr().Val().L1() = get_matrixLine(Ori,0);
    aOriConique.Externe().ParamRotation().CodageMatr().Val().L2() = get_matrixLine(Ori,1);
    aOriConique.Externe().ParamRotation().CodageMatr().Val().L3() = get_matrixLine(Ori,2);
    double xEstimate = aOriConique.Externe().Centre().x;
    double yEstimate = aOriConique.Externe().Centre().y;
    double zEstimate = aOriConique.Externe().Centre().z;
    for (unsigned int i=0;i<PoseToInit.size();i++)
    {
        cout<<"   -- ";
        //calculate positon pose
        aOriConique.Externe().Centre().x = xEstimate;
        aOriConique.Externe().Centre().y = yEstimate;
        aOriConique.Externe().Centre().z = zEstimate;
        xEstimate = xEstimate + vectorAvancement.x;
        yEstimate = yEstimate + vectorAvancement.y;
        zEstimate = zEstimate + vectorAvancement.z;
        //make file XML
        MakeFileXML(aOriConique, "Ori-"+aOriOut+"/Orientation-"+PoseToInit[i]+".xml");
        cout<<xEstimate<<" "<<yEstimate<<" "<<zEstimate<<endl;
    }
}
void OrientationLinear (vector<string> PoseToInit, Pt3dr vectorAvancement, cOrientationConique OriRef, string aOriOut)
{
    cout<<"Init with vector: "<<vectorAvancement<<endl;
    cOrientationConique aOriConique = OriRef;
    double xEstimate = aOriConique.Externe().Centre().x;
    double yEstimate = aOriConique.Externe().Centre().y;
    double zEstimate = aOriConique.Externe().Centre().z;
    for (unsigned int i=0;i<PoseToInit.size();i++)
    {
        cout<<"   -- ";
        //calculate positon pose
        aOriConique.Externe().Centre().x = xEstimate;
        aOriConique.Externe().Centre().y = yEstimate;
        aOriConique.Externe().Centre().z = zEstimate;
        xEstimate = xEstimate + vectorAvancement.x;
        yEstimate = yEstimate + vectorAvancement.y;
        zEstimate = zEstimate + vectorAvancement.z;
        //make file XML
        MakeFileXML(aOriConique, "Ori-"+aOriOut+"/Orientation-"+PoseToInit[i]+".xml");
        cout<<xEstimate<<" "<<yEstimate<<" "<<zEstimate<<endl;
    }
}

Pt3dr CalVecAvancementInit (vector<string> PoseRef, string aOriRef)
{
    cout<<"Calculate vector d'avancement : "<<endl;
    double xBefore=0, yBefore=0, zBefore=0;
    double xAcc = 0, yAcc = 0, zAcc = 0;
    vector<string> aSetRefImages = PoseRef;
    for (unsigned int i=0;i<aSetRefImages.size();i++)
    {   //tout les poses references dans camera
        std::cout<<"  - "<<aSetRefImages[i]<<" ";
        std::string aOriRefImage="Ori-"+aOriRef+"/Orientation-"+aSetRefImages[i]+".xml";
        cOrientationConique aOriConique=StdGetFromPCP(aOriRefImage,OrientationConique); //prendre orientation Conique partie a partir de XML fichier
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
    Pt3dr result(xMov,yMov, zMov);
    cout<<endl<<"Init with vector movement = "<<xMov<<" ; "<<yMov<<" ; "<<zMov<<" ; "<<endl;
    return result;
}

void RotationParAxe(Pt3dr AxeDirection, double angle, matrix result)
{
    double c=cos(angle);
    double s=sin(angle);
    double module=sqrt(pow(AxeDirection.x,2) + pow(AxeDirection.y,2) + pow(AxeDirection.z,2));
    Pt3dr AxeDirectionUnit = AxeDirection.operator /(module);
    result[0][0] = pow(AxeDirectionUnit.x,2)*(1-c)+c;
    result[0][1] = (AxeDirectionUnit.x * AxeDirectionUnit.y)*(1-c)-AxeDirectionUnit.z*s;
    result[0][2] = (AxeDirectionUnit.x * AxeDirectionUnit.z)*(1-c)+AxeDirectionUnit.y*s;

    result[1][0] = (AxeDirectionUnit.x * AxeDirectionUnit.y)*(1-c)+AxeDirectionUnit.z*s;
    result[1][1] = pow(AxeDirectionUnit.y,2)*(1-c)+c;
    result[1][2] = (AxeDirectionUnit.y * AxeDirectionUnit.z)*(1-c)-AxeDirectionUnit.x*s;

    result[2][0] = (AxeDirectionUnit.x * AxeDirectionUnit.z)*(1-c)-AxeDirectionUnit.y*s;
    result[2][1] = (AxeDirectionUnit.y * AxeDirectionUnit.z)*(1-c)+AxeDirectionUnit.x*s;
    result[2][2] = pow(AxeDirectionUnit.z,2)*(1-c)+c;

}

struct Section
{
    vector<string> Poses;
    double angle;
    bool isReference;
};
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
    vector<std::string> aVecPatternNewImages;
    vector<std::string> aVecPatternRefImages;
    std::string aFullPatternNewImages, aFullPatternRefImages, aOriRef,
            aPatternCam1, aPatternCam2, aPatternCam3, aPatternCam4,
            aPatternRef1, aPatternRef2, aPatternRef3, aPatternRef4;//pattern of all files
    std::string aVecPatternNewImages_E, aVecPatternRefImages_E, aPatPoseTurn, aPatAngle, aAxeOrient;
    aAxeOrient = "x";
    string aOriOut = "InitOut";
    bool bWithOriIdentity = false;
    bool forceInPlan = false;
    ElInitArgMain			//initialize Elise, set which is mandantory arg and which is optional arg
    (
    argc,argv,
    //mandatory arguments
    LArgMain()  << EAMC(aOriRef,"Reference Orientation Ori folder",  eSAM_IsExistDirOri)
                << EAMC(aOriOut, "Output initialized ori folder", eSAM_None)
                << EAMC(aVecPatternNewImages_E, "Vector pattern of new images to orientate Pat Cam 1, Pat Cam 2,..", eSAM_None)
                << EAMC(aVecPatternRefImages_E, "Vector pattern of Reference Image Pat Ref 1, Pat Ref 2,..", eSAM_None),
    //optional arguments
    LArgMain()  <<EAM(aPatPoseTurn, "PatTurn", true, "Vector of images when the serie change acquisition direction pose1,pose2...")
                <<EAM(aPatAngle,    "PatAngle", true, "Vector of turn angle apha1,alpha2,...")
                <<EAM(bWithOriIdentity,    "WithIdent", true, "Initialize with orientation identique (default = false)")
                <<EAM(aAxeOrient,    "Axe", true, "Which axe to calcul rotation about")
                <<EAM(forceInPlan,   "Plan", true, "Force using vector [0,0,1] to initialize (garantie all poses will be in a same plan) - (default = false)")
    );
    if (MMVisualMode) return EXIT_SUCCESS;

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


/*
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
    }*/

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
