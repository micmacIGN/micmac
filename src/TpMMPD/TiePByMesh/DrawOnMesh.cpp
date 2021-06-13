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

#include "InitOutil.h"
#include "../../uti_phgrm/NewOri/NewOri.h"
#include "DrawOnMesh.h"

Pt3dr DrawOnMesh::InterFaisce_cpy_cPI_Appli  (
                                               const std::vector<CamStenope *> & aVCS,
                                               const std::vector<Pt2dr> & aNPts2D
                                             )
{
    std::vector<ElSeg3D> aVSeg;
    for (int aKR=0 ; aKR < int(aVCS.size()) ; aKR++)
    {
        ElSeg3D aSeg = aVCS.at(aKR)->F2toRayonR3(aNPts2D.at(aKR));
        aVSeg.push_back(aSeg);
    }
    Pt3dr aRes =  ElSeg3D::L2InterFaisceaux(0,aVSeg,0);
    return aRes;
}

DrawOnMesh::DrawOnMesh(InitOutil *aChain)
{
    this->mChain = aChain;
    this->mPtrListPic = aChain->getmPtrListPic();
    this->mPtrListTri = aChain->getmPtrListTri();
}

DrawOnMesh::DrawOnMesh()
{}



vector<Pt3dr> DrawOnMesh::drawPt3DByPt2DAndTriangle(
                                                    triangle *tri,
                                                    vector<PtInteretInTriangle> pts2DInTri,
                                                    pic* img
                                                    )
{
    vector<Pt2dr> ptsInteret2DInTriOnImg;
    //select all pts of the same image
    for (uint i=0; i<pts2DInTri.size(); i++)
    {
        if (pts2DInTri[i].imageContainPtAndTriangle->mNameImg == img->mNameImg)
            { ptsInteret2DInTriOnImg.push_back(pts2DInTri[i].aPoint);}
    }
    vector<Pt3dr> result = tri->ptsInTri2Dto3D(ptsInteret2DInTriOnImg, img);
    return result;
}

Pt3dr DrawOnMesh::drawPt3DByInterPts2DManyImgs(
                                            vector<Pt2dr> pts2DonDifferentImgs,
                                            vector<pic*> imgContainPts2D
                                           )
{
    Pt3dr pts3D ;
    vector<CamStenope *> camImg;
    if (pts2DonDifferentImgs.size() > 1)
    {
        for (uint i=0; i<pts2DonDifferentImgs.size()-1; i++)
        {
          camImg.push_back(imgContainPts2D[i]->mOriPic);
        }
        pts3D = this->InterFaisce_cpy_cPI_Appli(camImg,pts2DonDifferentImgs ) ; //utilise Mehdi appli
    }
//    if (pts2DonDifferentImgs.size() > 1)
//    {
//    for (uint i=0; i<pts2DonDifferentImgs.size()-1; i++)
//        {
//            pic * img1 = imgContainPts2D[i];
//            pic * img2 = imgContainPts2D[i+1];
//            Pt2dr pts1 = pts2DonDifferentImgs[i];
//            Pt2dr pts2 = pts2DonDifferentImgs[i+1];
//            pts3D = img1->OriPic->ElCamera::PseudoInter(pts1, *(img2->OriPic), pts2);
//        }
//    }
    else
    {cout<<"Don't have enough pts to verify by intersect"<<endl;}
    return pts3D;
}

vector<Pt3dr> DrawOnMesh::drawPackHomoOnMesh(   ElPackHomologue aPack,
                                                pic* pic1, pic* pic2,
                                                Pt3dr color = Pt3dr(0,255,0) , string suffix=""
                                            )
{
    vector<Pt3dr> resultPts3d;
    CamStenope * cam1 = pic1->mOriPic;
    CamStenope * cam2 = pic2->mOriPic;
    for (ElPackHomologue::const_iterator itP=aPack.begin(); itP!=aPack.end() ; itP++)
    {
        Pt3dr pts3D= cam1->ElCamera::PseudoInter(itP->P1(), *cam2, itP->P2());
        resultPts3d.push_back(pts3D);
    }
    string aPlyOutDir;
    aPlyOutDir=mChain->getPrivmICNM()->Dir()+"PlyVerify/";
    if(!(ELISE_fp::IsDirectory(aPlyOutDir)))
        ELISE_fp::MkDir(aPlyOutDir);
    string aPlyFileName = pic1->getNameImgInStr() + pic2->getNameImgInStr() + suffix + "_HOMOPack.ply";
    creatPLYPts3D(resultPts3d, aPlyOutDir+aPlyFileName, Pt3dr(0,255,255));
    return resultPts3d;
}



void DrawOnMesh::creatPLYPts3D(vector<Pt3dr> pts3DAllTri,
                               vector<Pt3dr> listFace3D,
                               string fileName,
                               Pt3dr colorRGB)
{
      ofstream myfile;
      myfile.open (fileName.c_str());

      myfile<<"ply\n";
      myfile<<"format ascii 1.0\n";
      myfile<<"element vertex ";myfile<<pts3DAllTri.size();myfile<<"\n";
      myfile<<"property float x\n";
      myfile<<"property float y\n";
      myfile<<"property float z\n";
      myfile<<"property uchar red\n";
      myfile<<"property uchar green\n";
      myfile<<"property uchar blue\n";
      if (listFace3D.size() > 0)
      {
        myfile<<"element face "<<listFace3D.size()<<"\n";
        myfile<<"property list uint8 int32 vertex_index\n";
        myfile<<"property uchar red\n";
        myfile<<"property uchar green\n";
        myfile<<"property uchar blue\n";
      }
      myfile<<"end_header\n";
      for (double i=0; i<pts3DAllTri.size(); i++)
      {
          myfile<<pts3DAllTri[i].x;myfile<<" ";
          myfile<<pts3DAllTri[i].y;myfile<<" ";
          myfile<<pts3DAllTri[i].z;myfile<<" ";
          myfile<<colorRGB.x;myfile<<" ";
          myfile<<colorRGB.y;myfile<<" ";
          myfile<<colorRGB.z;myfile<<"\n";
      }
      for (double i=0; i<listFace3D.size(); i++)
      {
          myfile<<listFace3D[i].x;myfile<<" ";
          myfile<<listFace3D[i].y;myfile<<" ";
          myfile<<listFace3D[i].z;myfile<<" ";
          myfile<<colorRGB.x;myfile<<" ";
          myfile<<colorRGB.y;myfile<<" ";
          myfile<<colorRGB.z;myfile<<"\n";
      }
      myfile.close();
}

void DrawOnMesh::creatPLYPts3D(vector<Pt3dr> pts3DAllTri,
                               string fileName,
                               Pt3dr colorRGB)
{
      ofstream myfile;
      myfile.open (fileName.c_str());

      myfile<<"ply\n";
      myfile<<"format ascii 1.0\n";
      myfile<<"element vertex ";myfile<<pts3DAllTri.size();myfile<<"\n";
      myfile<<"property float x\n";
      myfile<<"property float y\n";
      myfile<<"property float z\n";
      myfile<<"property uchar red\n";
      myfile<<"property uchar green\n";
      myfile<<"property uchar blue\n";
      myfile<<"end_header\n";
      for (double i=0; i<pts3DAllTri.size(); i++)
      {
          myfile<<pts3DAllTri[i].x;myfile<<" ";
          myfile<<pts3DAllTri[i].y;myfile<<" ";
          myfile<<pts3DAllTri[i].z;myfile<<" ";
          myfile<<colorRGB.x;myfile<<" ";
          myfile<<colorRGB.y;myfile<<" ";
          myfile<<colorRGB.z;myfile<<"\n";
      }
      myfile.close();
}

void DrawOnMesh::drawEdge_p1p2(vector<Pt3dr> &p1Vec,
                               vector<Pt3dr> &p2Vec,
                               string filename,
                               Pt3dr colorRGBVer, Pt3dr colorRGBEdge)
{
    ofstream myfile;
    myfile.open (filename.c_str());

    myfile<<"ply\n";
    myfile<<"format ascii 1.0\n";
    myfile<<"element vertex ";myfile<<p1Vec.size()+p2Vec.size();myfile<<"\n";
    myfile<<"property float x\n";
    myfile<<"property float y\n";
    myfile<<"property float z\n";
    myfile<<"property uchar red\n";
    myfile<<"property uchar green\n";
    myfile<<"property uchar blue\n";
    myfile<<"element edge ";myfile<<p1Vec.size();myfile<<"\n";
    myfile<<"property int vertex1\n";
    myfile<<"property int vertex2\n";
    myfile<<"property uchar red\n";
    myfile<<"property uchar green\n";
    myfile<<"property uchar blue\n";
    myfile<<"end_header\n";

    for (double i=0; i<p1Vec.size(); i++)
    {
        myfile<<p1Vec[i].x;myfile<<" ";myfile<<p1Vec[i].y;myfile<<" ";myfile<<p1Vec[i].z;myfile<<" ";
        myfile<<colorRGBVer.x;myfile<<" ";
        myfile<<colorRGBVer.y;myfile<<" ";
        myfile<<colorRGBVer.z;myfile<<"\n";
        myfile<<p2Vec[i].x;myfile<<" ";myfile<<p2Vec[i].y;myfile<<" ";myfile<<p2Vec[i].z;myfile<<" ";
        myfile<<colorRGBVer.x;myfile<<" ";
        myfile<<colorRGBVer.y;myfile<<" ";
        myfile<<colorRGBVer.z;myfile<<"\n";
    }
    double count = 0;
    for (double i=0; i<p1Vec.size(); i++)
    {
        myfile<<count;myfile<<" ";count++;myfile<<count<<" ";count++;
        myfile<<colorRGBEdge.x;myfile<<" ";
        myfile<<colorRGBEdge.y;myfile<<" ";
        myfile<<colorRGBEdge.z;myfile<<"\n";
    }
    myfile.close();
}

void DrawOnMesh::drawListTriangle(  vector<triangle*> listTri,
                                    string fileName,
                                    Pt3dr colorRGB
                                  )
{
    ofstream myfile;
    myfile.open (fileName.c_str());

    myfile<<"ply\n";
    myfile<<"format ascii 1.0\n";
    myfile<<"element vertex ";myfile<<listTri.size()*3;myfile<<"\n";
    myfile<<"property float x\n";
    myfile<<"property float y\n";
    myfile<<"property float z\n";
    myfile<<"property uchar red\n";
    myfile<<"property uchar green\n";
    myfile<<"property uchar blue\n";
    myfile<<"element face "<<listTri.size()<<"\n";
    myfile<<"property list uint8 int32 vertex_index\n";
    myfile<<"property uchar red\n";
    myfile<<"property uchar green\n";
    myfile<<"property uchar blue\n";
    myfile<<"end_header\n";
    for (double i=0; i<listTri.size(); i++)
    {
        triangle * aTri = listTri[i];
        myfile<<aTri->getSommet(0).x;myfile<<" ";
        myfile<<aTri->getSommet(0).y;myfile<<" ";
        myfile<<aTri->getSommet(0).z;myfile<<" ";
        myfile<<colorRGB.x;myfile<<" ";
        myfile<<colorRGB.y;myfile<<" ";
        myfile<<colorRGB.z;myfile<<"\n";
        myfile<<aTri->getSommet(1).x;myfile<<" ";
        myfile<<aTri->getSommet(1).y;myfile<<" ";
        myfile<<aTri->getSommet(1).z;myfile<<" ";
        myfile<<colorRGB.x;myfile<<" ";
        myfile<<colorRGB.y;myfile<<" ";
        myfile<<colorRGB.z;myfile<<"\n";
        myfile<<aTri->getSommet(2).x;myfile<<" ";
        myfile<<aTri->getSommet(2).y;myfile<<" ";
        myfile<<aTri->getSommet(2).z;myfile<<" ";
        myfile<<colorRGB.x;myfile<<" ";
        myfile<<colorRGB.y;myfile<<" ";
        myfile<<colorRGB.z;myfile<<"\n";
    }
    double count = 0;
    for (double i=0; i<listTri.size(); i++)
    {
        myfile<<"3 ";
        myfile<<count;myfile<<" ";count++;
        myfile<<count<<" ";count++;
        myfile<<count<<" ";count++;
        myfile<<colorRGB.x;myfile<<" ";
        myfile<<colorRGB.y;myfile<<" ";
        myfile<<colorRGB.z;myfile<<"\n";
    }
    myfile.close();
}

void DrawOnMesh::drawListTriangle(vector<cXml_Triangle3DForTieP> listTri,
                                    string fileName,
                                    Pt3dr colorRGB
                                  )
{
    ofstream myfile;
    if (listTri.size() > 0)
        myfile.open (fileName.c_str());
    else
        return;

    myfile<<"ply\n";
    myfile<<"format ascii 1.0\n";
    myfile<<"element vertex ";myfile<<listTri.size()*3;myfile<<"\n";
    myfile<<"property float x\n";
    myfile<<"property float y\n";
    myfile<<"property float z\n";
    myfile<<"property uchar red\n";
    myfile<<"property uchar green\n";
    myfile<<"property uchar blue\n";
    myfile<<"element face "<<listTri.size()<<"\n";
    myfile<<"property list uint8 int32 vertex_index\n";
    myfile<<"property uchar red\n";
    myfile<<"property uchar green\n";
    myfile<<"property uchar blue\n";
    myfile<<"end_header\n";
    for (double i=0; i<listTri.size(); i++)
    {
        cXml_Triangle3DForTieP aTri = listTri[i];
        myfile<<aTri.P1().x;myfile<<" ";
        myfile<<aTri.P1().y;myfile<<" ";
        myfile<<aTri.P1().z;myfile<<" ";
        myfile<<colorRGB.x;myfile<<" ";
        myfile<<colorRGB.y;myfile<<" ";
        myfile<<colorRGB.z;myfile<<"\n";
        myfile<<aTri.P2().x;myfile<<" ";
        myfile<<aTri.P2().y;myfile<<" ";
        myfile<<aTri.P2().z;myfile<<" ";
        myfile<<colorRGB.x;myfile<<" ";
        myfile<<colorRGB.y;myfile<<" ";
        myfile<<colorRGB.z;myfile<<"\n";
        myfile<<aTri.P3().x;myfile<<" ";
        myfile<<aTri.P3().y;myfile<<" ";
        myfile<<aTri.P3().z;myfile<<" ";
        myfile<<colorRGB.x;myfile<<" ";
        myfile<<colorRGB.y;myfile<<" ";
        myfile<<colorRGB.z;myfile<<"\n";
    }
    double count = 0;
    for (double i=0; i<listTri.size(); i++)
    {
        myfile<<"3 ";
        myfile<<count;myfile<<" ";count++;
        myfile<<count<<" ";count++;
        myfile<<count<<" ";count++;
        myfile<<colorRGB.x;myfile<<" ";
        myfile<<colorRGB.y;myfile<<" ";
        myfile<<colorRGB.z;myfile<<"\n";
    }
    myfile.close();
}

void DrawOnMesh::drawListTriangle(vector<vector<Pt3dr> > listTri,
                                    string fileName,
                                    Pt3dr colorRGB
                                 )
{
    ofstream myfile;
    if (listTri.size() > 0)
        myfile.open (fileName.c_str());
    else
        return;

    myfile<<"ply\n";
    myfile<<"format ascii 1.0\n";
    myfile<<"element vertex ";myfile<<listTri.size()*3;myfile<<"\n";
    myfile<<"property float x\n";
    myfile<<"property float y\n";
    myfile<<"property float z\n";
    myfile<<"property uchar red\n";
    myfile<<"property uchar green\n";
    myfile<<"property uchar blue\n";
    myfile<<"element face "<<listTri.size()<<"\n";
    myfile<<"property list uint8 int32 vertex_index\n";
    myfile<<"property uchar red\n";
    myfile<<"property uchar green\n";
    myfile<<"property uchar blue\n";
    myfile<<"end_header\n";
    for (double i=0; i<listTri.size(); i++)
    {
        vector<Pt3dr> aTri = listTri[i];
        myfile<<aTri[0].x;myfile<<" ";
        myfile<<aTri[0].y;myfile<<" ";
        myfile<<aTri[0].z;myfile<<" ";
        myfile<<colorRGB.x;myfile<<" ";
        myfile<<colorRGB.y;myfile<<" ";
        myfile<<colorRGB.z;myfile<<"\n";
        myfile<<aTri[1].x;myfile<<" ";
        myfile<<aTri[1].y;myfile<<" ";
        myfile<<aTri[1].z;myfile<<" ";
        myfile<<colorRGB.x;myfile<<" ";
        myfile<<colorRGB.y;myfile<<" ";
        myfile<<colorRGB.z;myfile<<"\n";
        myfile<<aTri[2].x;myfile<<" ";
        myfile<<aTri[2].y;myfile<<" ";
        myfile<<aTri[2].z;myfile<<" ";
        myfile<<colorRGB.x;myfile<<" ";
        myfile<<colorRGB.y;myfile<<" ";
        myfile<<colorRGB.z;myfile<<"\n";
    }
    double count = 0;
    for (double i=0; i<listTri.size(); i++)
    {
        myfile<<"3 ";
        myfile<<count;myfile<<" ";count++;
        myfile<<count<<" ";count++;
        myfile<<count<<" ";count++;
        myfile<<colorRGB.x;myfile<<" ";
        myfile<<colorRGB.y;myfile<<" ";
        myfile<<colorRGB.z;myfile<<"\n";
    }
    myfile.close();
}


void DrawOnMesh::drawListPtsOnPly(
                                    vector<Pt3dr> lstPts,
                                    string filename,
                                    Pt3dr colorRGB
                                    )
{
    creatPLYPts3D(lstPts, filename, colorRGB);
}


