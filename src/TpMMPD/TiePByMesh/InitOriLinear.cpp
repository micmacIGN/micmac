#include "InitOriLinear.h"
#include <stdio.h>

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
   result.L1()  = Pt3dr(fin_mat[0][0], fin_mat[0][1], fin_mat[0][2]);
   result.L2()  = Pt3dr(fin_mat[1][0], fin_mat[1][1], fin_mat[1][2]);
   result.L3()  = Pt3dr(fin_mat[2][0], fin_mat[2][1], fin_mat[2][2]);
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



SerieCamLinear::SerieCamLinear(string aPatImgREF, string aPatImgNEW,
                               string aOri, string aOriOut,
                               string aAxeOrient, vector<double> aMulF, int index)
{
    this->mOriOut = aOriOut;
    this->mIndexCam = index;
    this->mPatImgNEW = aPatImgNEW;
    this->mPatImgREF = aPatImgREF;
    this->mOri = aOri;
    this->mMulF = aMulF;
    this->mAxeOrient = aAxeOrient;
    string aDirNEW,aDirREF, aPatNEW,aPatREF;
    SplitDirAndFile(aDirNEW, aPatNEW, mPatImgNEW);
    SplitDirAndFile(aDirREF, aPatREF, mPatImgREF);
    this->mICNM=cInterfChantierNameManipulateur::BasicAlloc(aDirNEW);
    this->mSetImgNEW = *(mICNM->Get(mPatImgNEW));
    this->mSetImgREF = *(mICNM->Get(mPatImgREF));
    string aDirOri;
    for (uint i=0; i<mSetImgREF.size(); i++)
    {
        aDirOri= mOri + "/Orientation-"+mSetImgREF[i]+".xml";
        bool Exist = ELISE_fp::exist_file(aDirOri);
        if (!Exist)
            StdCorrecNameOrient(mOri,aDirREF);
        aDirOri= mOri + "/Orientation-"+mSetImgREF[i]+".xml";
        cOrientationConique aOriFile=StdGetFromPCP(aDirOri,OrientationConique);
        this->mSetOriREF.push_back(aOriFile);
    }
    cout<<"Cam has: "<<endl;
    cout<<" ++ "<<mSetImgREF.size()<<" imgs REF"<<endl;
    for (uint i=0; i<mSetImgREF.size(); i++)
        cout<<"     ++"<<mSetImgREF[i]<<endl;
    cout<<" ++ "<<mSetImgNEW.size()<<" imgs NEW"<<endl;
    for (uint i=0; i<mSetImgNEW.size(); i++)
        cout<<"     ++"<<mSetImgNEW[i]<<endl;
}


void SerieCamLinear::calPosRlt()
{
    cout<<"Cal Position Relative of Cam "<<this->mIndexCam<<endl;
    if (mSystem.size() == 0)
        cout<<"ERROR : mSystem havent' had any cam !"<<endl;
    else
    {
        cOrientationConique aOriREFImg0ThisCam = this->mSetOriREF[0];
        for(uint i=0; i<mSystem.size(); i++)
        {
          SerieCamLinear * cam = mSystem[i];
          cOrientationConique aOriREFImg0OtherCam = cam->mSetOriREF[0];
          Pt3dr posRlt = aOriREFImg0OtherCam.Externe().Centre() - aOriREFImg0ThisCam.Externe().Centre();
          this->posRltWithOtherCam.push_back(posRlt);
          cout<<" ++Cam "<<cam->mIndexCam<<" : "<<posRlt<<endl;
        }
    }
}

Pt3dr SerieCamLinear::calVecMouvement()
{
    cout<<"Cal Vector Deplacement Cam "<<this->mIndexCam;
    Pt3dr result;
    cOrientationConique aOriImg0 = mSetOriREF[0];
    Pt3dr CentreImg0 = aOriImg0.Externe().Centre();
    Pt3dr acc(0,0,0);
    for (uint i=1; i<this->mSetOriREF.size(); i++)
    {
        cOrientationConique aOriImg = mSetOriREF[i];
        Pt3dr CentreImg = aOriImg.Externe().Centre();
        acc = acc + CentreImg - CentreImg0;
        CentreImg0 = CentreImg;
    }
    result = acc/(this->mSetOriREF.size()-1);
    this->mVecMouvement = result;
    cout<<" "<<mVecMouvement<<endl;
    return result;
}

void SerieCamLinear::initSerie(Pt3dr vecMouvCam0 ,
                               vector<string> aVecPoseTurn,
                               vector<double> aVecAngleTurn)
{
    cout<<"Init serie Cam : "<<this->mIndexCam<<endl;
    if (aVecPoseTurn.size() == 0)
    {
        cOrientationConique aOriLastImg = this->mSetOriREF.back();
        for (uint i=0; i<this->mSetImgNEW.size(); i++)
        {
            cOrientationConique aOriInitImg = aOriLastImg;
            aOriInitImg.Externe().Centre() = aOriLastImg.Externe().Centre() + this->mVecMouvement;
            string aOriInitImgXML = this->mOriOut + "/Orientation-"+this->mSetImgNEW[i]+".xml";
            MakeFileXML(aOriInitImg, aOriInitImgXML);
            aOriLastImg = aOriInitImg;
            cout<<" ++ "<<aOriInitImg.Externe().Centre()<<endl;
            cout<<" ++ Write: "<<aOriInitImgXML<<endl;
            this->mSetOriNEW.push_back(aOriInitImg);
        }
    }
}

void SerieCamLinear::partageSection(vector<string> aVecPoseTurn,
                                    vector<double> aVecAngleTurn)
{
        int k=0;
        vector<string> setImgNEW = this->mSetImgNEW;
        for (uint i=0; i<aVecPoseTurn.size(); i++)
        {
            vector<string> aSection;
            string imgTurn = aVecPoseTurn[i];
            for (uint j=k; j<setImgNEW.size(); j++)
            {
                if (setImgNEW[j] != imgTurn)
                    aSection.push_back(setImgNEW[j]);
                else
                {
                    aSection.push_back(setImgNEW[j]);
                    k = j+1;
                    break;
                }
            }
            this->mSections.push_back(aSection);
        }
        for (uint i=0; i<mSections.size(); i++)
        {
            cout<<"Section "<<i<<" : "<<endl;
            for (uint j=0; j<mSections[i].size(); j++)
            {
                cout<<" ++"<<mSections[i][j]<<endl;
            }
        }
}

void SerieCamLinear::calCodageMatrRot(cTypeCodageMatr ref, double angle,
                                      cTypeCodageMatr &  out, string axe)
{
    matrix org;
    matrix orient;
    matrix outMat;
    set_matrixLine(org, ref.L1() ,0);
    set_matrixLine(org, ref.L2() ,1);
    set_matrixLine(org, ref.L3() ,2);
    cout<<" Matrix rotation axe "<<axe<<" "<<angle<< " rad"<<endl;
    Pt3dr line0, line1, line2;
    cout<<"["<<org[0][0]<<","<<org[0][1]<<","<<org[0][2]<<"]"<<endl
        <<"["<<org[1][0]<<","<<org[1][1]<<","<<org[1][2]<<"]"<<endl
        <<"["<<org[2][0]<<","<<org[2][1]<<","<<org[2][2]<<"]"<<endl;
    if (axe == "x")
    {
        line0 = Pt3dr (1, 0 , 0);
        line1 = Pt3dr (0,cos(angle) ,-sin(angle) );
        line2 = Pt3dr (0,sin(angle) ,cos(angle) );
    }
    if (axe == "y")
    {
        line0 = Pt3dr(cos(angle),0 , sin(angle));
        line1 = Pt3dr(0,1,0 );
        line2 = Pt3dr(-sin(angle),0,cos(angle) );
    }
    if (axe == "z")
    {
        line0 = Pt3dr(cos(angle) ,-sin(angle), 0);
        line1 = Pt3dr(sin(angle) ,cos(angle),0 );
        line2 = Pt3dr(0,0,1 );
    }
    set_matrixLine(orient, line0 ,0);
    set_matrixLine(orient, line1 ,1);
    set_matrixLine(orient, line2 ,2);
    mult_matrix(orient,org, outMat);
    out.L1() = Pt3dr(outMat[0][0],outMat[0][1],outMat[0][2]);
    out.L2() = Pt3dr(outMat[1][0],outMat[1][1],outMat[1][2]);
    out.L3() = Pt3dr(outMat[2][0],outMat[2][1],outMat[2 ][2]);
    cout<<"["<<outMat[0][0]<<","<<outMat[0][1]<<","<<outMat[0][2]<<"]"<<endl
        <<"["<<outMat[1][0]<<","<<outMat[1][1]<<","<<outMat[1][2]<<"]"<<endl
        <<"["<<outMat[2][0]<<","<<outMat[2][1]<<","<<outMat[2][2]<<"]"<<endl;
}

Pt3dr SerieCamLinear::calVecMouvementTurn(Pt3dr vecRef, double angle, string axe)
{
    matrix orient;
    Pt3dr line0, line1, line2;
    if (axe == "x")
    {
        line0 = Pt3dr(1,0 , 0);
        line1 = Pt3dr (0,cos(angle) ,-sin(angle) );
        line2 = Pt3dr (0,sin(angle) ,cos(angle) );
    }
    if (axe == "y")
    {
        line0 = Pt3dr(cos(angle),0 , sin(angle));
        line1 = Pt3dr(0,1,0 );
        line2 = Pt3dr(-sin(angle),0,cos(angle) );
    }
    if (axe == "z")
    {
        line0 = Pt3dr(cos(angle) ,-sin(angle), 0);
        line1 = Pt3dr(sin(angle) ,cos(angle),0 );
        line2 = Pt3dr(0,0,1 );
    }
    set_matrixLine(orient, line0 ,0);
    set_matrixLine(orient, line1 ,1);
    set_matrixLine(orient, line2 ,2);
    return mult_vector(orient,vecRef);
}


void SerieCamLinear::initSerieWithTurn(     Pt3dr vecMouvCam0 ,
                                            vector<string> aVecPoseTurn,
                                            vector<double> aVecAngleTurn    )
{
    if (this->mMulF.size() == aVecAngleTurn.size())
    {
        cout<<"Init serie with turn Cam : "<<this->mIndexCam<<endl;
        if (aVecPoseTurn.size() != 0)
        {
            cOrientationConique aOriREFLastImg = this->mSetOriREF.back();
            for (uint i=0; i<this->mSections.size(); i++)
            {
                cout<<"REF Section:"<<aOriREFLastImg.Externe().ParamRotation().CodageMatr().Val().L1()
                                    <<aOriREFLastImg.Externe().ParamRotation().CodageMatr().Val().L2()
                                    <<aOriREFLastImg.Externe().ParamRotation().CodageMatr().Val().L3()<<endl;
                vector<string> poseInSection = mSections[i];
                //section 0
                if (i==0) //section 0 is same direction with reference pose
                {
                    for (uint j=0; j<poseInSection.size(); j++)
                    {
                        cOrientationConique aOriInitImg = aOriREFLastImg;
                        aOriInitImg.Externe().Centre() = aOriREFLastImg.Externe().Centre() + this->mVecMouvement;
                        string aOriInitImgXML = this->mOriOut + "/Orientation-"+poseInSection[j]+".xml";
                        MakeFileXML(aOriInitImg, aOriInitImgXML);
                        aOriREFLastImg = aOriInitImg;
                        //cout<<" ++ "<<aOriInitImg.Externe().Centre()<<endl;
                        //cout<<" ++ Write: "<<aOriInitImgXML<<endl;
                        this->mSetOriNEW.push_back(aOriInitImg);
                    }
                }
                //autre section
                else
                {
                    cTypeCodageMatr ref(aOriREFLastImg.Externe().ParamRotation().CodageMatr().Val());
                    cTypeCodageMatr out;
                    cout<<" Mat Rot org= "<<ref.L1()<<ref.L2()<<ref.L3()<<endl;
                    calCodageMatrRot(ref, aVecAngleTurn[i-1]*PI/180, out ,this->mAxeOrient);
                    Pt3dr vecMouvSection = calVecMouvementTurn(this->mVecMouvement, aVecAngleTurn[i-1]*PI/180, this->mAxeOrient);
                    Pt3dr vecAdjLastPos = this->mVecMouvement;

                    cout<<" Mat Rot sec= "<<out.L1()<<out.L2()<<out.L3()<<endl;
                    for (uint j=0; j<poseInSection.size(); j++)
                    {
                        cOrientationConique aOriInitImg = aOriREFLastImg;
                        if (j==0)
                        {
                            aOriInitImg.Externe().Centre() = aOriREFLastImg.Externe().Centre() + vecMouvSection*mMulF[i-1] + vecAdjLastPos*mMulF[i-1];
                            vecAdjLastPos = vecMouvSection;
                        }
                        else
                            aOriInitImg.Externe().Centre() = aOriREFLastImg.Externe().Centre() + vecMouvSection;
                        aOriInitImg.Externe().ParamRotation().CodageMatr().Val() = out;
                        string aOriInitImgXML = this->mOriOut + "/Orientation-"+poseInSection[j]+".xml";
                        MakeFileXML(aOriInitImg, aOriInitImgXML);
                        aOriREFLastImg = aOriInitImg;
                        //cout<<" ++ "<<aOriInitImg.Externe().Centre()<<endl;
                        //cout<<" ++ Write: "<<aOriInitImgXML<<endl;
                        this->mSetOriNEW.push_back(aOriInitImg);
                    }
                    this->mVecMouvement = vecMouvSection;
                }
            }
        }
    }
    else
    {
        cout<<"mMulF != aVecAngleTurn - QUIT"<<endl;
    }
}

void SerieCamLinear::initSerieByRefSerie(SerieCamLinear* REFSerie)
{
    cout<<"Init serie Cam : "<<this->mIndexCam<<endl;
    Pt3dr vecPosRlt = REFSerie->posRltWithOtherCam[this->mIndexCam];
    vector<cOrientationConique> aSetOriNEWSerieRef = REFSerie->mSetOriNEW;
    for (uint i=0; i<this->mSetImgNEW.size(); i++)
    {
        cOrientationConique oriRef = aSetOriNEWSerieRef[i];
        cOrientationConique oriNew = oriRef;
        oriNew.Externe().Centre() = oriRef.Externe().Centre() + vecPosRlt;
        string aOriInitImgXML = this->mOriOut + "/Orientation-"+this->mSetImgNEW[i]+".xml";
        MakeFileXML(oriNew, aOriInitImgXML);
        cout<<" ++ "<<oriNew.Externe().Centre()<<endl;
        cout<<" ++ Write: "<<aOriInitImgXML<<endl;
        this->mSetOriNEW.push_back(oriNew);
    }
}


