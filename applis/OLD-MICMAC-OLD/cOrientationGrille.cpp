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
#include "general/all.h"
#include <iostream>
#include <vector>
#include <string>
#include <complex>
#include <fstream>
#include <iomanip>
#include <list>
#include <stdexcept>

#include "cModuleOrientation.h"
#include "cOrientationGrille.h"

#ifdef __AVEC_XERCES__
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>

XERCES_CPP_NAMESPACE_USE
#else
#ifdef VERSION
#undef VERSION
#endif
#include "private/files.h"
#endif

// variable static de la classe
std::map<std::string, OrientationGrille*> OrientationGrille::dico_camera;

void OrientationGrille::WriteBinary(std::string const &nom)const
{
	std::cout << "OrientationGrille::WriteBinary"<<std::endl;
	std::ofstream fic(nom.c_str(),ios::binary);
        double pt2dD[2];
        int pt2dI[2];
	pt2dD[0]=Image2Obj_ULC.real();
	pt2dD[1]=Image2Obj_ULC.imag();
        fic.write((const char*)&pt2dD,2*sizeof(double));
        pt2dD[0]=Image2Obj_Pas.real();
        pt2dD[1]=Image2Obj_Pas.imag();
	fic.write((const char*)&pt2dD,2*sizeof(double));
        pt2dI[0]=Image2Obj_Taille.real();
        pt2dI[1]=Image2Obj_Taille.imag();
	fic.write((const char*)&pt2dI,2*sizeof(int));
	fic.write((const char*)&Image2Obj_TailleZ,sizeof(int));
        int NbCouches = (int) Image2Obj.size();
        fic.write((const char*)&NbCouches,sizeof(int));
	size_t Taille = Image2Obj_Taille.real()*Image2Obj_Taille.imag()*Image2Obj_TailleZ*sizeof(double);
        for(int i=0;i<NbCouches;++i)
	{
		std::vector<double> const &layer=Image2Obj[i];
		fic.write((const char*)&Image2Obj_Value[i],sizeof(double));
		fic.write((const char*)&(layer[0]),(unsigned int) Taille);
	}
	pt2dD[0]=Obj2Image_ULC.real();
        pt2dD[1]=Obj2Image_ULC.imag();
        fic.write((const char*)&pt2dD,2*sizeof(double));
        pt2dD[0]=Obj2Image_Pas.real();
        pt2dD[1]=Obj2Image_Pas.imag();
        fic.write((const char*)&pt2dD,2*sizeof(double));
        pt2dI[0]=Obj2Image_Taille.real();
        pt2dI[1]=Obj2Image_Taille.imag();
        fic.write((const char*)&pt2dI,2*sizeof(int));
        fic.write((const char*)&Obj2Image_TailleZ,sizeof(int));
        NbCouches = (int) Obj2Image.size();
        fic.write((const char*)&NbCouches,sizeof(int));
        Taille = Obj2Image_Taille.real()*Obj2Image_Taille.imag()*Obj2Image_TailleZ*sizeof(double);
        for(int i=0;i<NbCouches;++i)
        {
                std::vector<double> const &layer=Obj2Image[i];
                fic.write((const char*)&Obj2Image_Value[i],sizeof(double));
                fic.write((const char*)&(layer[0]),(unsigned int) Taille);
        }
	fic.close();
}

void OrientationGrille::InitBinary(std::string const &nom)
{
	std::cout << "OrientationGrille::InitBinary"<<std::endl;
	camera=NULL;
	std::ifstream fic(nom.c_str(),ios::binary);
	double pt2dD[2];
	int pt2dI[2];
	fic.read((char*)&pt2dD,2*sizeof(double));
	Image2Obj_ULC=std::complex<double>(pt2dD[0],pt2dD[1]);
	fic.read((char*)&pt2dD,2*sizeof(double));
        Image2Obj_Pas=std::complex<double>(pt2dD[0],pt2dD[1]);
	fic.read((char*)&pt2dI,2*sizeof(int));
        Image2Obj_Taille=std::complex<int>(pt2dI[0],pt2dI[1]);
	fic.read((char*)&Image2Obj_TailleZ,sizeof(int));
	int NbCouches;
	fic.read((char*)&NbCouches,sizeof(int));
	Image2Obj.resize(NbCouches);
	Image2Obj_Value.resize(NbCouches);
	size_t Taille = Image2Obj_Taille.real()*Image2Obj_Taille.imag()*Image2Obj_TailleZ*sizeof(double);
	for(int i=0;i<NbCouches;++i)
	{
		std::vector<double> &layer=Image2Obj[i];
		layer.resize(Taille);
		fic.read((char*)&Image2Obj_Value[i],sizeof(double));
		fic.read((char*)&(layer[0]),(unsigned int) Taille);
	}
	fic.read((char*)&pt2dD,2*sizeof(double));
        Obj2Image_ULC=std::complex<double>(pt2dD[0],pt2dD[1]);
        fic.read((char*)&pt2dD,2*sizeof(double));
        Obj2Image_Pas=std::complex<double>(pt2dD[0],pt2dD[1]);
        fic.read((char*)&pt2dI,2*sizeof(int));
        Obj2Image_Taille=std::complex<int>(pt2dI[0],pt2dI[1]);
        fic.read((char*)&Obj2Image_TailleZ,sizeof(int));
        fic.read((char*)&NbCouches,sizeof(int));
        Obj2Image.resize(NbCouches);
	Obj2Image_Value.resize(NbCouches);
	Taille = Obj2Image_Taille.real()*Obj2Image_Taille.imag()*Obj2Image_TailleZ*sizeof(double);
        for(int i=0;i<NbCouches;++i)
        {
                std::vector<double> &layer=Obj2Image[i];
                layer.resize(Taille);
		fic.read((char*)&Obj2Image_Value[i],sizeof(double));
                fic.read((char*)&(layer[0]),(unsigned int) Taille);
        }
	fic.close();
}

OrientationGrille::OrientationGrille(std::string const &nom):ModuleOrientation(nom)
{
	Image2Obj_Taille = std::complex<int>(0,0);
	Image2Obj_TailleZ = 0;
	Obj2Image_Taille = std::complex<int>(0,0);
	Obj2Image_TailleZ = 0;

	camera = NULL;

	bool isXML = true;
	{
		std::ifstream fic(nom.c_str());
		if (!fic.good())
		{
			std::cout << "Fichier : "<<nom<<" non valide"<<std::endl;
			throw std::logic_error("[OrientationGrille::OrientationGrille] : Erreur dans la lecture du fichier");
		}
		char c;
		fic >> c;
		isXML = (c == '<');
	}
	if (isXML)
		InitXML(nom);
	else 
		InitBinary(nom);

	if (Obj2Image.size()==0)
        {
                std::cout << "Grille inverse non disponible" << std::endl;

                // exit (-1);
// TRAITEMENT PROVISOIRE MPD A CORRIGER
/*
                MicMacErreur
                (
                    eErrGrilleInverseNonDisponible,
                    "La grille de contient pas de grille inverse",
                     ""
                );

*/
//
                throw std::logic_error("[OrientationGrille::OrientationGrille] : Grille inverse non disponible");
        }
	
	PrecisionRetour = GetResolMoyenne()*0.1;	
}

void OrientationGrille::InitXML(std::string const &nom)
{
	std::cout << "Debut de lecture du ficier XML : " << nom << std::endl;
	std::string nomCamera("");
#ifdef __AVEC_XERCES__
	try {
		XMLPlatformUtils::Initialize();
	}
	catch (const XMLException& toCatch) {
		char* message = XMLString::transcode(toCatch.getMessage());
		std::cout << "Error during initialization! :\n"
			<< message << "\n";
		XMLString::release(&message);
		return;
	}
	XercesDOMParser* parser = new XercesDOMParser();
	parser->setValidationScheme(XercesDOMParser::Val_Always); // optional 
	parser->setDoNamespaces(true); // optional 
	ErrorHandler* errHandler = (ErrorHandler*) new HandlerBase();
	parser->setErrorHandler(errHandler);
	try {
		parser->parse(nom.c_str());
	}
	catch (const XMLException& toCatch) {
		char* message = XMLString::transcode(toCatch.getMessage());
		std::cout << "Exception message is: \n"
			<< message << "\n";
		XMLString::release(&message);
		return;
	}
	catch (const DOMException& toCatch) {
		char* message = XMLString::transcode(toCatch.msg);
		std::cout << "Exception message is: \n"
			<< message << "\n";
		XMLString::release(&message);
		return;
	}
	catch (...) {
		std::cout << "Unexpected Exception \n" ;
		return;
	}
	DOMNode* doc = parser->getDocument();
	DOMNode* n = doc->getFirstChild();
	if (n)
		n=n->getFirstChild();
	while(n)
	{
		if (!XMLString::compareString(n->getNodeName(),XMLString::transcode("trans_coord")))
                {
                        DOMNode* sn = n->getFirstChild();
                        while(sn)
                        {
                                if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("sys_coord")))
                                {
                                        DOMNamedNodeMap* att = sn->getAttributes();
                                        if (!att)
                                                continue;
                                        if (XMLString::compareString(att->getNamedItem(XMLString::transcode("name"))->getNodeValue(),XMLString::transcode("sys1")))
                                                continue;
                                        DOMNode* sn2 = sn->getFirstChild();
                                        while(sn2)
                                        {
                                                if (!XMLString::compareString(sn2->getNodeName(),XMLString::transcode("sys_coord_plani")))
                                                {
                                                        DOMNode* sn3 = sn2->getFirstChild();
                                                        while(sn3)
                                                        {
                                                                if (!XMLString::compareString(sn3->getNodeName(),XMLString::transcode("sub_code")))
                                                                {
                                                                        nomCamera=std::string(XMLString::transcode(sn3->getFirstChild()->getNodeValue()));
                                                                        sn3=NULL;
                                                                }
                                                                else sn3=sn3->getNextSibling();
                                                        }
                                                        sn2 = NULL;
                                                }
                                                else sn2=sn2->getNextSibling();
                                        }
                                }
                                sn=sn->getNextSibling();
                        }
                }
                else if (!XMLString::compareString(n->getNodeName(),XMLString::transcode("multi_grid")))
		{
			std::cout << "ICI" << std::endl;
			DOMNamedNodeMap* att = n->getAttributes();
			if (!att)
				continue;
			std::complex<double> *Pas;
			std::complex<int> *Taille;
			std::complex<double> *ULC;
			std::vector<std::vector<double> > *Grille;
			std::vector<double> *Value;
			int *TZ;
			if (!XMLString::compareString(att->getNamedItem(XMLString::transcode("name"))->getNodeValue(),XMLString::transcode("2-1")))
			{
				std::cout << "Gri 2->1"<<std::endl;
				Pas = &Obj2Image_Pas;
				Taille = &Obj2Image_Taille;
				TZ = &Obj2Image_TailleZ;
				ULC = &Obj2Image_ULC;
				Grille = &Obj2Image;
				Value = &Obj2Image_Value;
			}
			else
			{
				std::cout << "Gri 1->2"<<std::endl;
				Pas = &Image2Obj_Pas;
				Taille = &Image2Obj_Taille;
				TZ = &Image2Obj_TailleZ;
				ULC = &Image2Obj_ULC;
				Grille = &Image2Obj;
				Value = &Image2Obj_Value;
			}
			DOMNode* sn = n->getFirstChild();
			while(sn)
			{
				if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("columns_interval")))
				{
					(*Pas)=std::complex<double>(atof(XMLString::transcode(sn->getFirstChild()->getNodeValue())),Pas->imag());
				}
				else if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("rows_interval")))
				{
					(*Pas)=std::complex<double>(Pas->real(),atof(XMLString::transcode(sn->getFirstChild()->getNodeValue())));
				}
				if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("columns_number")))
				{
					(*Taille)=std::complex<int>(atoi(XMLString::transcode(sn->getFirstChild()->getNodeValue())),Taille->imag());
				}
				else if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("rows_number")))
				{
					(*Taille)=std::complex<int>(Taille->real(),atoi(XMLString::transcode(sn->getFirstChild()->getNodeValue())));
				}
				else if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("components_number")))
				{
					(*TZ)=atoi(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
				}
				else if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("upper_left")))
				{
					std::istringstream buffer(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
					double ULCx,ULCy;
					buffer >> ULCx >> ULCy;
					(*ULC) = std::complex<double>(ULCx,ULCy);
				}
				else if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("layer")))
				{
					//DOMNamedNodeMap* snatt = sn->getAttributes();
					double value = atof(XMLString::transcode(sn->getAttributes()->getNamedItem(XMLString::transcode("value"))->getNodeValue()));
					std::cout << "value : "<<value<<std::endl;
					Value->push_back(value);
					Grille->push_back(std::vector<double>());
					std::vector<double> & grille = (*Grille)[Grille->size()-1];
					std::istringstream buffer(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
					int T = Taille->real()*Taille->imag()*(*TZ);
					grille.resize(T);
					for(int i=0;i<T;++i)
						buffer >> grille[i];
				}
				sn=sn->getNextSibling();
			}
		}

		n = n->getNextSibling();
	}
	delete parser;
	delete errHandler;
#else
	cElXMLTree tree(nom);
	// Recherche de la camera eventuelle
        {
                std::list<cElXMLTree*> noeuds=tree.GetAll(std::string("sys_coord_plani"));
                std::list<cElXMLTree*>::iterator it_grid,fin_grid=noeuds.end();
                for(it_grid=noeuds.begin();it_grid!=fin_grid;++it_grid)
                {
                        std::string name = (*it_grid)->ValAttr("name");
                        if (name==std::string("sys1"))
                        {
                                cElXMLTree* pt = (*it_grid)->GetUnique("sys_coord_plani");
                                if (!pt->GetUnique("sub_code")->IsVide())
                                {
                                        nomCamera=pt->GetUnique("sub_code")->GetUniqueVal();
                                }
                        }
                }
        }
	// Lecture des grilles
        {
        std::list<cElXMLTree*> noeuds=tree.GetAll(std::string("multi_grid"));
	std::list<cElXMLTree*>::iterator it_grid,fin_grid=noeuds.end();
	for(it_grid=noeuds.begin();it_grid!=fin_grid;++it_grid)
	{
		std::string name = (*it_grid)->ValAttr("name");
		std::cout << "MULTI_GRID : "<<name<<std::endl;

		std::complex<double> *Pas;
		std::complex<int> *Taille;
		std::complex<double> *ULC;
		std::vector<std::vector<double> > *Grille;
		std::vector<double> *Value;
		int *TZ;

		if (name==std::string("2-1"))
		{
			std::cout << "Gri 2->1"<<std::endl;
			Pas = &Obj2Image_Pas;
			Taille = &Obj2Image_Taille;
			TZ = &Obj2Image_TailleZ;
			ULC = &Obj2Image_ULC;
			Grille = &Obj2Image;
			Value = &Obj2Image_Value;
		}
		else
		{
			std::cout << "Gri 1->2"<<std::endl;
			Pas = &Image2Obj_Pas;
			Taille = &Image2Obj_Taille;
			TZ = &Image2Obj_TailleZ;
			ULC = &Image2Obj_ULC;
			Grille = &Image2Obj;
			Value = &Image2Obj_Value;
		}

		(*Pas)= std::complex<double>((*it_grid)->GetUniqueValDouble("columns_interval"),(*it_grid)->GetUniqueValDouble("rows_interval"));
		(*Taille) = std::complex<int>((*it_grid)->GetUniqueValInt("columns_number"),(*it_grid)->GetUniqueValInt("rows_number"));
		(*TZ) = (*it_grid)->GetUniqueValInt("components_number");
		//std::cout << "Taille : "<<(*Taille)<<std::endl;
		//std::cout << "Pas : "<<(*Pas)<<std::endl;
		// Lecture du ULCorner
		{
			double ULCx,ULCy;
			std::istringstream buffer((*it_grid)->GetUnique("upper_left")->GetUniqueVal());
			buffer >> ULCx >> ULCy;
			(*ULC)=std::complex<double>(ULCx,ULCy);
		}
		//std::cout << "ULC : "<<(*ULC)<<std::endl;
		// Lecuture des layers
		{
			std::list<cElXMLTree*> layers=(*it_grid)->GetAll("layer");
			std::list<cElXMLTree*>::iterator it,fin=layers.end();
			for(it=layers.begin();it!=fin;++it)
			{
				double value=atof((*it)->ValAttr("value").c_str());
				std::istringstream buffer((*it)->Contenu());
				int T = Taille->real()*Taille->imag()*(*TZ);
				Grille->push_back(std::vector<double>());
				Value->push_back(value);
				std::vector<double> &layer=(*Grille)[Grille->size()-1];
				layer.resize(T);	
				for(int c=0;c<T;++c)
				{
					buffer >> layer[c];
				}
			}		
		}
		std::cout << "Nombre de couches : "<< (unsigned int) Value->size()<<std::endl;
	}
	}
#endif
	std::cout << "Camera : "<<nomCamera<<std::endl;
        if ((nomCamera.length()>0)&&(nomCamera!=std::string("*")))
        {
                std::map<std::string, OrientationGrille*>::iterator it=dico_camera.find(nomCamera);
                if (it==dico_camera.end())
                {
                        // recuperation du chemin de fichier 
                        std::string path;
                        {
                                int placeSlash = -1;
                                for(int l=nom.size()-1;(l>=0)&&(placeSlash==-1);--l)
                                {
                                        if ( ( nom[l]=='/' )||( nom[l]=='\\' ) )
                                        {
                                                placeSlash = l;
                                        }
                                }

                                path = std::string("");
                                if (placeSlash!=-1)
                                {
                                        path.assign(nom.begin(),nom.begin()+placeSlash+1);
                                }
                        }

                        std::string nomFichierCamera = path+nomCamera+std::string(".gri");
                        std::cout << "Chargement d'un nouveau fichier de camera : "<<nomFichierCamera<<std::endl;
                        camera = new OrientationGrille(nomFichierCamera);
                        dico_camera.insert(std::pair<std::string,OrientationGrille*>(nomCamera,camera));
                }
                else
                {
                        std::cout << "Utilisation d'une camera deja chargee"<<std::endl;
                        camera = (*it).second;
                }
        }
	std::cout << "Fin du Chargement" << std::endl;
}

void OrientationGrille::ImageAndPx2Obj(double c, double l, const double *aPx,
		double &x, double &y)const
{
        std::complex<double> position_grille;
        // Si besoin on corrige la distorsion
        if (camera)
        {
                double c_sans_distorsion,l_sans_distorsion = l;
                camera->ImageAndPx2Obj(c,l,aPx,c_sans_distorsion,l_sans_distorsion);
                // Position dans la grille
                position_grille = std::complex<double>((c_sans_distorsion-Image2Obj_ULC.real())/Image2Obj_Pas.real(),
                                (Image2Obj_ULC.imag()-l_sans_distorsion)/Image2Obj_Pas.imag());
        }
        else
        {
                // Position dans la grille
                position_grille = std::complex<double>((c-Image2Obj_ULC.real())/Image2Obj_Pas.real(),
                                (Image2Obj_ULC.imag()-l)/Image2Obj_Pas.imag());
        }
        // Si on n'a qu'une couche
        if (Image2Obj_Value.size() == 1)
        {
                std::complex<double> P = interpolation(position_grille,Image2Obj[0],Image2Obj_Taille,Image2Obj_TailleZ);
                x = P.real();
                y = P.imag();
        }
        else
        {
                // Recherche des deux niveaux les plus proches
                int l1=-1;
                double d1=0.;
                int l2=-1;
                double d2=0.;
                for(size_t id=0;id<Image2Obj_Value.size();++id)
                {
                        double d=aPx[0]-Image2Obj_Value[id];
                        if (l1==-1)
                        {
                                l1 = (int) id;
                                d1 = d;
                        }
                        else if (std::abs(d)<std::abs(d1))
                        {
                                l2 = l1;
                                d2 = -d1;
                                l1 = (int) id;
                                d1 = d;
                        }
                        else if ((l2==-1)||(std::abs(d)<std::abs(d2)))
                        {
                                l2 = (int) id;
                                d2 = -d;
                        }
                }

                std::complex<double> P1 = interpolation(position_grille,Image2Obj[l1],Image2Obj_Taille,Image2Obj_TailleZ);
                std::complex<double> P2 = interpolation(position_grille,Image2Obj[l2],Image2Obj_Taille,Image2Obj_TailleZ);

                x = (P1.real()*d2+P2.real()*d1)/(d1+d2);
                y = (P1.imag()*d2+P2.imag()*d1)/(d1+d2);
        }
}

void OrientationGrille::Objet2ImageInit(double x, double y, const double *aPx,
		double &c, double &l)const
{
        // On verifie que la grille est disponible
        if (Obj2Image.size()==0)
        {
                std::cout << "Grille inverse non disponible" << std::endl;
		throw std::logic_error("[OrientationGrille::Objet2ImageInit] : Grille inverse non disponible");
                return;
        }
        // Position dans la grille
        std::complex<double> position_grille((x-Obj2Image_ULC.real())/Obj2Image_Pas.real(),
                        (Obj2Image_ULC.imag()-y)/Obj2Image_Pas.imag());
        // Si il n'y a qu'un seul niveau
        if (Obj2Image_Value.size()==1)
        {
                std::complex<double> P = interpolation(position_grille,Obj2Image[0],Obj2Image_Taille,Obj2Image_TailleZ);

                c = P.real();
                l = P.imag();
        }
        else
        {
                // Recherche des deux niveaux les plus proches
                int l1=-1;
                double d1=0.;
                int l2=-1;
                double d2=0.;
                for(size_t id=0;id<Obj2Image_Value.size();++id)
                {
                        double d=aPx[0]-Obj2Image_Value[id];
                        if (l1==-1)
                        {
                                l1 = (int) id;
                                d1 = d;
                        }
                        else if (std::abs(d)<std::abs(d1))
                        {
                                l2 = l1;
                                d2 = -d1;
                                l1 = (int) id;
                                d1 = d;
                        }
                        else if ((l2==-1)||(std::abs(d)<std::abs(d2)))
                        {
                                l2 = (int) id;
                                d2 = -d;
                        }
                }

                std::complex<double> P1 = interpolation(position_grille,Obj2Image[l1],Obj2Image_Taille,Obj2Image_TailleZ);
                std::complex<double> P2 = interpolation(position_grille,Obj2Image[l2],Obj2Image_Taille,Obj2Image_TailleZ);

                c = (P1.real()*d2+P2.real()*d1)/(d1+d2);
                l = (P1.imag()*d2+P2.imag()*d1)/(d1+d2);
        }

        // Si besoin on applique la correction de distorsion
        if (camera)
        {
                // transfo de coord sans distorsion vers des coord avec prise en compte de la distorsion
                double c_sans_disto = c;
                double l_sans_disto = l;
                camera->Objet2ImageInit(c_sans_disto,l_sans_disto,aPx,c,l);
        }

	bool verbose = false;

        double Xretour,Yretour;
        ImageAndPx2Obj(c,l,aPx,Xretour,Yretour);
        
	if (verbose) std::cout << "Point Retour : "<<Xretour<<" "<<Yretour<<std::endl;

	double Err = sqrt((x-Xretour)*(x-Xretour)+(y-Yretour)*(y-Yretour));

	if (verbose) std::cout << "Err = "<<Err<<" / "<<PrecisionRetour<<std::endl;

        int NbEtape = 0;
        while((Err>PrecisionRetour)&&(NbEtape<10))
        {
		if (verbose) std::cout << "NbEtape : "<<NbEtape<<std::endl;
                ++NbEtape;
                double X1,Y1,X2,Y2;
                ImageAndPx2Obj(c+1,l,aPx,X1,Y1);
                ImageAndPx2Obj(c,l+1,aPx,X2,Y2);
                double dX1,dY1,dX2,dY2;
                dX1 = X1-Xretour;dY1 = Y1-Yretour;
                dX2 = X2-Xretour;dY2 = Y2-Yretour;
                double N = sqrt(dX1*dX1+dY1*dY1);
                if (N!=0.)
                {
                        double dc = (dX1*(Xretour-x)+dY1*(Yretour-y))/N/N;
                        double dl = (dX2*(Xretour-x)+dY2*(Yretour-y))/N/N;
                        c=c-dc;
                        l=l-dl;
                        ImageAndPx2Obj(c,l,aPx,Xretour,Yretour);
			if (verbose) std::cout << "Nouveau retour : "<<Xretour<<" "<<Yretour<<std::endl;
                        Err = sqrt((x-Xretour)*(x-Xretour)+(y-Yretour)*(y-Yretour));
			if (verbose) std::cout << "Nouvelle Err = "<<Err<<std::endl;
                }
                else Err = 0.;
        }
}

double OrientationGrille::GetResolMoyenne() const
{
	if (Image2Obj_Taille.real()<2)
		return 1.;
	// Estimation de la resolution au centre de la grille
	// Pos : la position du centre de l'image (de la Grille)
	int pos = Image2Obj_Taille.real()/2*Image2Obj_Taille.imag()/2;
	// Vect : le vecteur Obj entre deux points voisins (sur une ligne) au centre de la grille (les points : pos et (pos+1))
	std::complex<double> Vect(Image2Obj[0][2*(pos+1)  ]-Image2Obj[0][2*pos   ],
			Image2Obj[0][2*(pos+1)+1]-Image2Obj[0][2*pos +1]);
	// On divise la norme de ce vecteur (distance en geometrie Obj) par le pas de la grille en X (un nombre de pixels)
	double reso = sqrt(Vect.real()*Vect.real()+Vect.imag()*Vect.imag())/Image2Obj_Pas.real();
	return reso;
}

bool OrientationGrille::GetPxMoyenne(double * aPxMoy) const
{
	std::cout << "GetPxMoyenne" << std::endl;
	std::cout << "Attention on n'a pas d'info sur l'altitude moyenne du sol dans les fichiers GRI"<< std::endl;	
	aPxMoy[0]=0.;
	return true;
} 

std::complex<double> OrientationGrille::interpolation(std::complex<double> position_grille,std::vector<double> const &grille,std::complex<int> Taille, int TZ) const
{
	if ((Taille.real()<2)||(Taille.imag()<2)||(TZ<2))
		return std::complex<double>(0.,0.);
	int col = (int)floor(position_grille.real());
	int lig = (int)floor(position_grille.imag());
	if (col<0) col=0;
	if (lig<0) lig=0;
	if (col>(Taille.real()-2)) col = Taille.real()-2;
	if (lig>(Taille.imag()-2)) lig = Taille.imag()-2;
	double dcol = position_grille.real()-col;
	double dlig = position_grille.imag()-lig;
	// Les quatres Points autour de cette position
	std::complex<double> A(grille[(lig*Taille.real()+col)*TZ],
			grille[(lig*Taille.real()+col)*TZ+1]);
	std::complex<double> B(grille[(lig*Taille.real()+col+1)*TZ],
			grille[(lig*Taille.real()+col+1)*TZ+1]);
	std::complex<double> C(grille[((lig+1)*Taille.real()+col)*TZ],
			grille[((lig+1)*Taille.real()+col)*TZ+1]);
	std::complex<double> D(grille[((lig+1)*Taille.real()+col+1)*TZ],
			grille[((lig+1)*Taille.real()+col+1)*TZ+1]);
	// Interpolation bi lineaire
	std::complex<double> Pt((A.real()*(1-dcol)+B.real()*dcol)*(1-dlig)+(C.real()*(1-dcol)+D.real()*dcol)*dlig,
			(A.imag()*(1-dcol)+B.imag()*dcol)*(1-dlig)+(C.imag()*(1-dcol)+D.imag()*dcol)*dlig);
	return Pt;
}



/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
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
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/
