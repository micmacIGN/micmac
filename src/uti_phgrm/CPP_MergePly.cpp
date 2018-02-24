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
#include <unordered_set>

void writeHeader(FILE * aFP, int aNelems, int aType, bool aBin, const std::vector<std::string> &aComments, bool doublePrec)
{
    fprintf(aFP,"ply\n");
    string aBinSpec = MSBF_PROCESSOR() ? "binary_big_endian":"binary_little_endian" ;

    fprintf(aFP,"format %s 1.0\n",aBin?aBinSpec.c_str():"ascii");

    fprintf(aFP,"comment MergePly generated\n");
    for (unsigned int i=0;i<aComments.size();i++)
        fprintf(aFP,"comment %s\n",aComments[i].c_str());
    fprintf(aFP,"element vertex %d\n", aNelems);

    if (doublePrec)
    {
        fprintf(aFP,"property float64 x\n");
        fprintf(aFP,"property float64 y\n");
        fprintf(aFP,"property float64 z\n");
    }
    else
    {
        fprintf(aFP,"property float x\n");
        fprintf(aFP,"property float y\n");
        fprintf(aFP,"property float z\n");
    }

    switch (aType)
    {
        case 0:
            break;
        case 1:
        {
            fprintf(aFP,"property uchar red\n");
            fprintf(aFP,"property uchar green\n");
            fprintf(aFP,"property uchar blue\n");
            break;
        }
        case 2:
        {
            fprintf(aFP,"property uchar red\n");
            fprintf(aFP,"property uchar green\n");
            fprintf(aFP,"property uchar blue\n");
            fprintf(aFP,"property uchar alpha\n");
            break;
        }
        case 3:
        {
            fprintf(aFP,"property float nx\n");
            fprintf(aFP,"property float ny\n");
            fprintf(aFP,"property float nz\n");
            break;
        }
        case 4:
        {
            fprintf(aFP,"property float nx\n");
            fprintf(aFP,"property float ny\n");
            fprintf(aFP,"property float nz\n");
            fprintf(aFP,"property uchar red\n");
            fprintf(aFP,"property uchar green\n");
            fprintf(aFP,"property uchar blue\n");
            break;
        }
        case 5:
        {
            fprintf(aFP,"property float nx\n");
            fprintf(aFP,"property float ny\n");
            fprintf(aFP,"property float nz\n");
            fprintf(aFP,"property uchar red\n");
            fprintf(aFP,"property uchar green\n");
            fprintf(aFP,"property uchar blue\n");
            fprintf(aFP,"property uchar alpha\n");
            break;
        }
    }

    fprintf(aFP,"element face %d\n",0);
    fprintf(aFP,"property list uchar int vertex_indices\n");
    fprintf(aFP,"end_header\n");
}

int MergePly_main(int argc,char ** argv)
    {
        #ifdef _DEBUG
            cout << "using MergePly without QT" << endl;
        #endif

        string aFullName,aNameOut;
        string aDir, aPattern;
        vector<string> aVCom;

        int aBin  = 1;
        std::string aComment="";

        ElInitArgMain
        (
         argc,argv,
                    LArgMain()	<< EAMC(aFullName, "Full Name (Dir+Pattern)"),
         LArgMain()		<< EAM(aNameOut,"Out",true)
                        << EAM(aVCom,"Comments",true)
                        << EAM(aBin,"Bin",true,"Generate Binary or Ascii (Def=1, Binary)")
                        << EAM(aComment,"Comment",true,"Comment to add to header")
        );

        SplitDirAndFile(aDir, aPattern, aFullName);
        list<string> aVFiles = RegexListFileMatch(aDir, aPattern, 1, false);

        if (aNameOut=="")
            aNameOut = aDir + ELISE_CAR_DIR + StdPrefix(*(aVFiles.begin())) + "_merged.ply";

        //sPlyOrientedColoredAlphaVertex64 **glist=NULL;
        vector<sPlyOrientedColoredAlphaVertex64> vData64;
        vector<sPlyOrientedColoredAlphaVertex> vData;
        int gen_nelems =0;
        unordered_set<string> aComments;
        int Cptr = 0;

        int type = 0;
        bool wNormales = false;

        PlyFile * thePlyFile;
        int nelems, nprops, num_elems, file_type;
        float version;
        char **elist;
        char *elem_name;
        PlyProperty **plist=NULL;
        bool doublePrec = false;

        //get global number of elements
        list<string>::iterator itr = aVFiles.begin();
        for(;itr != aVFiles.end(); itr++)
        {
            thePlyFile = ply_open_for_reading( const_cast<char *>((aDir + ELISE_CAR_DIR + (*itr)).c_str()), &nelems, &elist, &file_type, &version);

            cout << "loading file " << *itr	<< endl;
//#ifdef _DEBUG
            cout << "version "	<< version		<< endl;
            cout << "type "		<< file_type	<< endl;
            cout << "nb elem "	<< nelems		<< endl;
//#endif

            elem_name = elist[0];
            plist = ply_get_element_description (thePlyFile, elem_name, &num_elems, &nprops);
            
            PlyProperty *p = plist[0];
            if ((!doublePrec) && (p->external_type == PLY_FLOAT_64))
            {
                cout << "l'attribut "<<p->name<<" du fichier "<<(*itr)<<" est stocke sur 64 bits, on passe en double precision"<<endl;
                doublePrec = true;
            }
            gen_nelems += num_elems;

            for (int i=0;i<thePlyFile->num_comments;i++)
                aComments.insert(thePlyFile->comments[i]);

            ply_close (thePlyFile);
        }

        cout << "nb total elem "	<< gen_nelems << endl;
        
        //read ply files
        itr = aVFiles.begin();
        for(;itr != aVFiles.end(); itr++)
        {
            thePlyFile = ply_open_for_reading( const_cast<char *>((aDir + ELISE_CAR_DIR +(*itr)).c_str()), &nelems, &elist, &file_type, &version);
            for (int i = 0; i < nelems; i++)
            {
                // get the description of the first element
                elem_name = elist[i];
                plist = ply_get_element_description (thePlyFile, elem_name, &num_elems, &nprops);
                if (!equal_strings ("vertex", elem_name)) continue;
                
                int external_type = plist[0]->external_type;
                
                switch(nprops)
                {
                    case 10: // x y z nx ny nz r g b a
                    {
                        type = 5;
                        PlyProperty props[] = {
                            {"x",  external_type, PLY_DOUBLE, offsetof(sPlyOrientedColoredAlphaVertex64,x ), 0, 0, 0, 0},
                            {"y",  external_type, PLY_DOUBLE, offsetof(sPlyOrientedColoredAlphaVertex64,y ), 0, 0, 0, 0},
                            {"z",  external_type, PLY_DOUBLE, offsetof(sPlyOrientedColoredAlphaVertex64,z ), 0, 0, 0, 0},
                            {"nx", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedColoredAlphaVertex64,nx), 0, 0, 0, 0},
                            {"ny", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedColoredAlphaVertex64,ny), 0, 0, 0, 0},
                            {"nz", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedColoredAlphaVertex64,nz), 0, 0, 0, 0},
                            {"red",   PLY_UCHAR, PLY_UCHAR, offsetof(sPlyOrientedColoredAlphaVertex64,red), 0, 0, 0, 0},
                            {"green", PLY_UCHAR, PLY_UCHAR, offsetof(sPlyOrientedColoredAlphaVertex64,green), 0, 0, 0, 0},
                            {"blue",  PLY_UCHAR, PLY_UCHAR, offsetof(sPlyOrientedColoredAlphaVertex64,blue), 0, 0, 0, 0},
                            {"alpha",  PLY_UCHAR, PLY_UCHAR, offsetof(sPlyOrientedColoredAlphaVertex64,alpha), 0, 0, 0, 0}
                        };
                        
                        for(int j=0;j<nprops;++j)
                        {
                            ply_get_property(thePlyFile, elem_name, &props[j]);
                        }
                        for (int j = 0; j < num_elems; j++, Cptr++)
                        {
                            sPlyOrientedColoredAlphaVertex64 vertex64;
                            ply_get_element (thePlyFile, (void *) &vertex64);
                            if (doublePrec)
                            {
                                vData64.push_back(vertex64);
                            }
                            else
                            {
                                sPlyOrientedColoredAlphaVertex vertex32;
                                vertex32.x = vertex64.x;
                                vertex32.y = vertex64.y;
                                vertex32.z = vertex64.z;
                                vertex32.nx = vertex64.nx;
                                vertex32.ny = vertex64.ny;
                                vertex32.nz = vertex64.nz;
                                vertex32.red   = vertex64.red;
                                vertex32.green = vertex64.green;
                                vertex32.blue  = vertex64.blue;
                                vertex32.alpha  = vertex64.alpha;
                                vData.push_back(vertex32);
                            }
                        }
                        break;
                    }
                    case 9: // x y z nx ny nz r g b
                    {
                        type = 4;
                        PlyProperty props[] = {
                            {"x",  external_type, PLY_DOUBLE, offsetof(sPlyOrientedColoredVertex64,x ), 0, 0, 0, 0},
                            {"y",  external_type, PLY_DOUBLE, offsetof(sPlyOrientedColoredVertex64,y ), 0, 0, 0, 0},
                            {"z",  external_type, PLY_DOUBLE, offsetof(sPlyOrientedColoredVertex64,z ), 0, 0, 0, 0},
                            {"nx", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedColoredVertex64,nx), 0, 0, 0, 0},
                            {"ny", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedColoredVertex64,ny), 0, 0, 0, 0},
                            {"nz", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedColoredVertex64,nz), 0, 0, 0, 0},
                            {"red",   PLY_UCHAR, PLY_UCHAR, offsetof(sPlyOrientedColoredVertex64,red), 0, 0, 0, 0},
                            {"green", PLY_UCHAR, PLY_UCHAR, offsetof(sPlyOrientedColoredVertex64,green), 0, 0, 0, 0},
                            {"blue",  PLY_UCHAR, PLY_UCHAR, offsetof(sPlyOrientedColoredVertex64,blue), 0, 0, 0, 0}
                        };
                        
                        for(int j=0;j<nprops;++j)
                        {
                            ply_get_property(thePlyFile, elem_name, &props[j]);
                        }
                        for (int j = 0; j < num_elems; j++, Cptr++)
                        {
                            sPlyOrientedColoredVertex64 vertex;
                            ply_get_element (thePlyFile, (void *) &vertex);
                            if (doublePrec)
                            {
                                sPlyOrientedColoredAlphaVertex64 vertex64;
                                vertex64.x = vertex.x;
                                vertex64.y = vertex.y;
                                vertex64.z = vertex.z;
                                vertex64.nx = vertex.nx;
                                vertex64.ny = vertex.ny;
                                vertex64.nz = vertex.nz;
                                vertex64.red   = vertex.red;
                                vertex64.green = vertex.green;
                                vertex64.blue  = vertex.blue;
                                vertex64.alpha  = 0;
                                vData64.push_back(vertex64);
                            }
                            else
                            {
                                sPlyOrientedColoredAlphaVertex vertex32;
                                vertex32.x = vertex.x;
                                vertex32.y = vertex.y;
                                vertex32.z = vertex.z;
                                vertex32.nx = vertex.nx;
                                vertex32.ny = vertex.ny;
                                vertex32.nz = vertex.nz;
                                vertex32.red   = vertex.red;
                                vertex32.green = vertex.green;
                                vertex32.blue  = vertex.blue;
                                vertex32.alpha  = 0;
                                vData.push_back(vertex32);
                            }
                        }
                        break;
                    }
                    case 7: // x y z r g b a
                    {
                        type = 2;
                        PlyProperty props[] = {
                            {"x",  external_type, PLY_DOUBLE, offsetof(sPlyColoredVertexWithAlpha64,x ), 0, 0, 0, 0},
                            {"y",  external_type, PLY_DOUBLE, offsetof(sPlyColoredVertexWithAlpha64,y ), 0, 0, 0, 0},
                            {"z",  external_type, PLY_DOUBLE, offsetof(sPlyColoredVertexWithAlpha64,z ), 0, 0, 0, 0},
                            {"red",   PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredVertexWithAlpha64,red), 0, 0, 0, 0},
                            {"green", PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredVertexWithAlpha64,green), 0, 0, 0, 0},
                            {"blue",  PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredVertexWithAlpha64,blue), 0, 0, 0, 0},
                            {"alpha",  PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredVertexWithAlpha64,alpha), 0, 0, 0, 0}
                        };
                        
                        for(int j=0;j<nprops;++j)
                        {
                            ply_get_property(thePlyFile, elem_name, &props[j]);
                        }
                        for (int j = 0; j < num_elems; j++, Cptr++)
                        {
                            sPlyColoredVertexWithAlpha64 vertex;
                            ply_get_element (thePlyFile, (void *) &vertex);
                            if (doublePrec)
                            {
                                sPlyOrientedColoredAlphaVertex64 vertex64;
                                vertex64.x = vertex.x;
                                vertex64.y = vertex.y;
                                vertex64.z = vertex.z;
                                vertex64.nx = 0.;
                                vertex64.ny = 0.;
                                vertex64.nz = 0.;
                                vertex64.red   = vertex.red;
                                vertex64.green = vertex.green;
                                vertex64.blue  = vertex.blue;
                                vertex64.alpha  = vertex.alpha;
                                vData64.push_back(vertex64);
                            }
                            else
                            {
                                sPlyOrientedColoredAlphaVertex vertex32;
                                vertex32.x = vertex.x;
                                vertex32.y = vertex.y;
                                vertex32.z = vertex.z;
                                vertex32.nx = 0.;
                                vertex32.ny = 0.;
                                vertex32.nz = 0.;
                                vertex32.red   = vertex.red;
                                vertex32.green = vertex.green;
                                vertex32.blue  = vertex.blue;
                                vertex32.alpha  = vertex.alpha;
                                vData.push_back(vertex32);
                            }
                        }
                        break;
                    }
                    case 6: // can be (x y z r g b) or (x y z nx ny nz)
                    {
                        for (int j = 0; j < nprops; j++)
                            if ( "nx"==plist[j]->name )   wNormales = true;
                        
                        if (!wNormales) // x y z r g b
                        {
                            type = 1;
                            PlyProperty props[] = {
                                {"x",  external_type, PLY_DOUBLE, offsetof(sPlyColoredVertex64,x ), 0, 0, 0, 0},
                                {"y",  external_type, PLY_DOUBLE, offsetof(sPlyColoredVertex64,y ), 0, 0, 0, 0},
                                {"z",  external_type, PLY_DOUBLE, offsetof(sPlyColoredVertex64,z ), 0, 0, 0, 0},
                                {"red",   PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredVertex64,red), 0, 0, 0, 0},
                                {"green", PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredVertex64,green), 0, 0, 0, 0},
                                {"blue",  PLY_UCHAR, PLY_UCHAR, offsetof(sPlyColoredVertex64,blue), 0, 0, 0, 0}
                            };
                            
                            for(int j=0;j<nprops;++j)
                            {
                                ply_get_property(thePlyFile, elem_name, &props[j]);
                            }
                            for (int j = 0; j < num_elems; j++, Cptr++)
                            {
                                sPlyColoredVertex64 vertex;
                                ply_get_element (thePlyFile, (void *) &vertex);
                                if (doublePrec)
                                {
                                    sPlyOrientedColoredAlphaVertex64 vertex64;
                                    vertex64.x = vertex.x;
                                    vertex64.y = vertex.y;
                                    vertex64.z = vertex.z;
                                    vertex64.nx = 0.;
                                    vertex64.ny = 0.;
                                    vertex64.nz = 0.;
                                    vertex64.red   = vertex.red;
                                    vertex64.green = vertex.green;
                                    vertex64.blue  = vertex.blue;
                                    vertex64.alpha  = 0;
                                    vData64.push_back(vertex64);
                                }
                                else
                                {
                                    sPlyOrientedColoredAlphaVertex vertex32;
                                    vertex32.x = vertex.x;
                                    vertex32.y = vertex.y;
                                    vertex32.z = vertex.z;
                                    vertex32.nx = 0.;
                                    vertex32.ny = 0.;
                                    vertex32.nz = 0.;
                                    vertex32.red   = vertex.red;
                                    vertex32.green = vertex.green;
                                    vertex32.blue  = vertex.blue;
                                    vertex32.alpha  = 0;
                                    vData.push_back(vertex32);
                                }
                            }
                        }
                        else // x y z nx ny nz
                        {
                            type = 3;
                            PlyProperty props[] = {
                                {"x",  external_type, PLY_DOUBLE, offsetof(sPlyOrientedVertex64,x ), 0, 0, 0, 0},
                                {"y",  external_type, PLY_DOUBLE, offsetof(sPlyOrientedVertex64,y ), 0, 0, 0, 0},
                                {"z",  external_type, PLY_DOUBLE, offsetof(sPlyOrientedVertex64,z ), 0, 0, 0, 0},
                                {"nx", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedVertex64,nx), 0, 0, 0, 0},
                                {"ny", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedVertex64,ny), 0, 0, 0, 0},
                                {"nz", PLY_FLOAT, PLY_FLOAT, offsetof(sPlyOrientedVertex64,nz), 0, 0, 0, 0}
                            };
                            
                            for(int j=0;j<nprops;++j)
                            {
                                ply_get_property(thePlyFile, elem_name, &props[j]);
                            }
                            for (int j = 0; j < num_elems; j++, Cptr++)
                            {
                                sPlyOrientedVertex64 vertex;
                                ply_get_element (thePlyFile, (void *) &vertex);
                                if (doublePrec)
                                {
                                    sPlyOrientedColoredAlphaVertex64 vertex64;
                                    vertex64.x = vertex.x;
                                    vertex64.y = vertex.y;
                                    vertex64.z = vertex.z;
                                    vertex64.nx = vertex.nx;
                                    vertex64.ny = vertex.ny;
                                    vertex64.nz = vertex.nz;
                                    vertex64.red   = 0;
                                    vertex64.green = 0;
                                    vertex64.blue  = 0;
                                    vertex64.alpha  = 0;
                                    vData64.push_back(vertex64);
                                }
                                else
                                {
                                    sPlyOrientedColoredAlphaVertex vertex32;
                                    vertex32.x = vertex.x;
                                    vertex32.y = vertex.y;
                                    vertex32.z = vertex.z;
                                    vertex32.nx = vertex.nx;
                                    vertex32.ny = vertex.ny;
                                    vertex32.nz = vertex.nz;
                                    vertex32.red   = 0;
                                    vertex32.green = 0;
                                    vertex32.blue  = 0;
                                    vertex32.alpha  = 0;
                                    vData.push_back(vertex32);
                                }
                            }
                        }
                        break;
                    }
                    case 3: // x y z
                    {
                        PlyProperty props[] = {
                            {"x",  external_type, PLY_DOUBLE, offsetof(sVertex64,x ), 0, 0, 0, 0},
                            {"y",  external_type, PLY_DOUBLE, offsetof(sVertex64,y ), 0, 0, 0, 0},
                            {"z",  external_type, PLY_DOUBLE, offsetof(sVertex64,z ), 0, 0, 0, 0}
                        };
                        
                        for(int j=0;j<nprops;++j)
                        {
                            ply_get_property(thePlyFile, elem_name, &props[j]);
                        }
                        for (int j = 0; j < num_elems; j++, Cptr++)
                        {
                            sVertex64 vertex;
                            ply_get_element (thePlyFile, (void *) &vertex);
                            if (doublePrec)
                            {
                                sPlyOrientedColoredAlphaVertex64 vertex64;
                                vertex64.x = vertex.x;
                                vertex64.y = vertex.y;
                                vertex64.z = vertex.z;
                                vertex64.nx = 0.;
                                vertex64.ny = 0.;
                                vertex64.nz = 0.;
                                vertex64.red   = 0;
                                vertex64.green = 0;
                                vertex64.blue  = 0;
                                vertex64.alpha  = 0;
                                vData64.push_back(vertex64);
                            }
                            else
                            {
                                sPlyOrientedColoredAlphaVertex vertex32;
                                vertex32.x = vertex.x;
                                vertex32.y = vertex.y;
                                vertex32.z = vertex.z;
                                vertex32.nx = 0.;
                                vertex32.ny = 0.;
                                vertex32.nz = 0.;
                                vertex32.red   = 0;
                                vertex32.green = 0;
                                vertex32.blue  = 0;
                                vertex32.alpha  = 0;
                                vData.push_back(vertex32);
                            }
                        }
                        break;
                    }
                    default:
                    {
                        printf("unable to load a ply unless number of properties is not 3, 6, 7, 9 or 10\n");
                        break;
                    }
                }
            }
            ply_close (thePlyFile);
        }

        //write ply file

        //Mode Ecriture : binaire ou non
        string mode = aBin ? "wb" : "w";
        FILE * aFP = FopenNN(aNameOut, mode, "MergePly");

        if (aComment.size()>0)
            aComments.insert(aComment);
        vector<string> aCommentsVector;
        aCommentsVector.assign(aComments.begin(),aComments.end());
        writeHeader(aFP, gen_nelems, type, aBin, aCommentsVector, doublePrec);

        //data
        if (doublePrec)
        {
            for (int aK=0 ; aK< gen_nelems ; aK++)
            {
                sPlyOrientedColoredAlphaVertex64 const& pt = vData64[aK];
                
                if (aBin)
                {
                    WriteType(aFP, pt.x);
                    WriteType(aFP, pt.y);
                    WriteType(aFP, pt.z);
                }
                
                switch (type)
                {
                    case 0:
                    {
                        if (!aBin)
                            fprintf(aFP,"%.7f %.7f %.7f\n", pt.x, pt.y, pt.z);
                        break;
                    }
                    case 1:
                    {
                        if (aBin)
                        {
                            WriteType(aFP, pt.red );
                            WriteType(aFP, pt.green );
                            WriteType(aFP, pt.blue );
                        }
                        else
                            fprintf(aFP,"%.7f %.7f %.7f %d %d %d\n",  pt.x, pt.y, pt.z, pt.red, pt.green, pt.blue);
                        
                        break;
                    }
                    case 2:
                    {
                        if (aBin)
                        {
                            WriteType(aFP, pt.red);
                            WriteType(aFP, pt.green);
                            WriteType(aFP, pt.blue);
                            WriteType(aFP, pt.alpha);
                        }
                        else
                            fprintf(aFP,"%.7f %.7f %.7f %d %d %d %d\n",  pt.x, pt.y, pt.z, pt.red, pt.green, pt.blue, pt.alpha);
                        break;
                    }
                    case 3:
                    {
                        if (aBin)
                        {
                            WriteType(aFP, pt.nx);
                            WriteType(aFP, pt.ny);
                            WriteType(aFP, pt.nz);
                        }
                        else
                            fprintf(aFP,"%.7f %.7f %.7f %.7f %.7f %.7f\n",  pt.x, pt.y, pt.z, pt.nx, pt.ny, pt.nz);
                        break;
                    }
                    case 4:
                    {
                        if (aBin)
                        {
                            WriteType(aFP, pt.nx);
                            WriteType(aFP, pt.ny);
                            WriteType(aFP, pt.nz);
                            WriteType(aFP, pt.red);
                            WriteType(aFP, pt.green);
                            WriteType(aFP, pt.blue);
                        }
                        else
                            fprintf(aFP,"%.7f %.7f %.7f %.7f %.7f %.7f %d %d %d\n",  pt.x, pt.y, pt.z, pt.nx, pt.y, pt.z, pt.red, pt.green, pt.blue );
                        break;
                    }
                    case 5:
                    {
                        if (aBin)
                        {
                            WriteType(aFP, pt.nx);
                            WriteType(aFP, pt.ny);
                            WriteType(aFP, pt.nz);
                            WriteType(aFP, pt.red);
                            WriteType(aFP, pt.green);
                            WriteType(aFP, pt.blue);
                            WriteType(aFP, pt.alpha);
                        }
                        else
                            fprintf(aFP,"%.7f %.7f %.7f %.7f %.7f %.7f %d %d %d %d\n",  pt.x, pt.y, pt.z, pt.x, pt.y, pt.z, pt.red, pt.green, pt.blue, pt.alpha );
                        break;
                    }
                }
            }
        }
        else
        {
            for (int aK=0 ; aK< gen_nelems ; aK++)
            {
                sPlyOrientedColoredAlphaVertex const& pt = vData[aK];
                
                if (aBin)
                {
                    WriteType(aFP, pt.x);
                    WriteType(aFP, pt.y);
                    WriteType(aFP, pt.z);
                }
                
                switch (type)
                {
                    case 0:
                    {
                        if (!aBin)
                            fprintf(aFP,"%.7f %.7f %.7f\n", pt.x, pt.y, pt.z);
                        break;
                    }
                    case 1:
                    {
                        if (aBin)
                        {
                            WriteType(aFP, pt.red );
                            WriteType(aFP, pt.green );
                            WriteType(aFP, pt.blue );
                        }
                        else
                            fprintf(aFP,"%.7f %.7f %.7f %d %d %d\n",  pt.x, pt.y, pt.z, pt.red, pt.green, pt.blue);
                        
                        break;
                    }
                    case 2:
                    {
                        if (aBin)
                        {
                            WriteType(aFP, pt.red);
                            WriteType(aFP, pt.green);
                            WriteType(aFP, pt.blue);
                            WriteType(aFP, pt.alpha);
                        }
                        else
                            fprintf(aFP,"%.7f %.7f %.7f %d %d %d %d\n",  pt.x, pt.y, pt.z, pt.red, pt.green, pt.blue, pt.alpha);
                        break;
                    }
                    case 3:
                    {
                        if (aBin)
                        {
                            WriteType(aFP, pt.nx);
                            WriteType(aFP, pt.ny);
                            WriteType(aFP, pt.nz);
                        }
                        else
                            fprintf(aFP,"%.7f %.7f %.7f %.7f %.7f %.7f\n",  pt.x, pt.y, pt.z, pt.nx, pt.ny, pt.nz);
                        break;
                    }
                    case 4:
                    {
                        if (aBin)
                        {
                            WriteType(aFP, pt.nx);
                            WriteType(aFP, pt.ny);
                            WriteType(aFP, pt.nz);
                            WriteType(aFP, pt.red);
                            WriteType(aFP, pt.green);
                            WriteType(aFP, pt.blue);
                        }
                        else
                            fprintf(aFP,"%.7f %.7f %.7f %.7f %.7f %.7f %d %d %d\n",  pt.x, pt.y, pt.z, pt.nx, pt.y, pt.z, pt.red, pt.green, pt.blue );
                        break;
                    }
                    case 5:
                    {
                        if (aBin)
                        {
                            WriteType(aFP, pt.nx);
                            WriteType(aFP, pt.ny);
                            WriteType(aFP, pt.nz);
                            WriteType(aFP, pt.red);
                            WriteType(aFP, pt.green);
                            WriteType(aFP, pt.blue);
                            WriteType(aFP, pt.alpha);
                        }
                        else
                            fprintf(aFP,"%.7f %.7f %.7f %.7f %.7f %.7f %d %d %d %d\n",  pt.x, pt.y, pt.z, pt.x, pt.y, pt.z, pt.red, pt.green, pt.blue, pt.alpha );
                        break;
                    }
                }
            }
        }
        

        ElFclose(aFP);

        if (plist!=NULL)
        {
            for(int j=0;j<nprops;++j)
            {
                free(plist[j]);
            }
            free(plist);
        }

        return EXIT_SUCCESS;
    }

/*Footer-MicMac-eLiSe-25/06/2007

 Ce logiciel est un programme informatique servant �  la mise en
 correspondances d'images pour la reconstruction du relief.

 Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
 respectant les principes de diffusion des logiciels libres. Vous pouvez
 utiliser, modifier et/ou redistribuer ce programme sous les conditions
 de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA
 sur le site "http://www.cecill.info".

 En contrepartie de l'accessibilité au code source et des droits de copie,
 de modification et de redistribution accordés par cette licence, il n'est
 offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
 seule une responsabilité restreinte pèse sur l'auteur du programme,  le
 titulaire des droits patrimoniaux et les concédants successifs.

 A cet égard  l'attention de l'utilisateur est attirée sur les risques
 associés au chargement,  �  l'utilisation,  �  la modification et/ou au
 développement et �  la reproduction du logiciel par l'utilisateur étant
 donné sa spécificité de logiciel libre, qui peut le rendre complexe �
 manipuler et qui le réserve donc �  des développeurs et des professionnels
 avertis possédant  des  connaissances  informatiques approfondies.  Les
 utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
 logiciel �  leurs besoins dans des conditions permettant d'assurer la
 sécurité de leurs systèmes et ou de leurs données et, plus généralement,
 �  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

 Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
 pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
 termes.
 Footer-MicMac-eLiSe-25/06/2007*/


