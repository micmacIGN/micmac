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


#include "../../uti_phgrm/NewOri/NewOri.h"
#include "InitOutil.h"
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
    this->mPtrListPic = aChain->getmPtrListPic();
    this->mPtrListTri = aChain->getmPtrListTri();
}

double DrawOnMesh::countPtsSurImg(pic* img)
{
    double count = 0;
    for (uint i=0; i<mPtrListTri.size(); i++)
    {
        vector<PtInteretInTriangle> databasePtsOnTri = mPtrListTri[i]->getPtsInteret2DInImagetteDuTri();
        for (uint j=0; j<databasePtsOnTri.size(); j++)
        {
               string nameImg = databasePtsOnTri[j].imageContainPtAndTriangle->getNameImgInStr();
               if ( nameImg.compare(img->getNameImgInStr()) == 0 )
               {count++;}
        }
    }
    return count;
}

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

vector<Pt3dr> DrawOnMesh::drawPackHomoOnMesh(
                                                ElPackHomologue aPack,
                                                pic* pic1, pic* pic2
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
    return resultPts3d;
}

void DrawOnMesh::drawTri3DAndAllPts3DInTri(triangle * tri, pic *img)
{
    //system("mkdir VerifPLY");
    //cout<<"Draw pts in tri "<<tri->index;
    //draw triangle 2D in 3D
//    Tri2d* aTri = tri->getReprSurImg()[img->index];
//    vector<Pt2dr> sommet;
//    sommet.push_back(aTri->sommet1[0]);
//    sommet.push_back(aTri->sommet1[1]);
//    sommet.push_back(aTri->sommet1[2]);
    vector<Pt3dr> aTri3D;
    aTri3D.push_back(tri->getSommet(0));
    aTri3D.push_back(tri->getSommet(1));
    aTri3D.push_back(tri->getSommet(2));
//    cout<<endl;
    //draw pts interet in tri by this img
    vector<Pt3dr> listPts3Dby1img =
            this->drawPt3DByPt2DAndTriangle (tri,
                                             tri->getPtsInteret2DInImagetteDuTri(),
                                             img);
    //cout<<endl;
    //draw pts interet in 3D by intersection

    //creat fichier PLY
    string fileName = "pts_" + intToString(tri->mIndex) + ".ply";
    creatPLYPts3D(listPts3Dby1img, fileName, Pt3dr(255,0,0));               // draw pts in triangle
    fileName = "tri_" + intToString(tri->mIndex) + ".ply";
    vector<Pt3dr> listFace3D;
    listFace3D.push_back(Pt3dr(0,1,2));
    creatPLYPts3D(aTri3D, listFace3D, fileName, Pt3dr(0,0,255));            //draw triangle
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





