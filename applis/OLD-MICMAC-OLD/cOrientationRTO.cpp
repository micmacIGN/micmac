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
#include "cOrientationRTO.h"


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
std::map<std::string, OrientationGrille*> OrientationRTO::dico_camera;

double Polynome::Valeur(double x, double y, double z) const
{
  double val = 0.;
  double pZ = 1.;
  std::vector<double>::const_iterator it = mdCoef.begin();
  for(int i=0;i<=mdDegZ;++i)
  {
    double pY = 1.;
    for(int j=0;j<=mdDegY;++j)
    {
      double pX = 1.;
      for(int k=0;k<=mdDegX;++k)
      {
        val += (*it) * pX * pY * pZ;
        ++it;
        pX*= x;
      }
      pY*= y;
    }
    pZ *= z;
  }
  return val;
}

void Polynome::Afficher()const
{
  std::vector<double>::const_iterator it = mdCoef.begin();
  for(int i=0;i<=mdDegZ;++i)
  {      
    for(int j=0;j<=mdDegY;++j)
    {
      for(int k=0;k<=mdDegX;++k)
      {
        std::cout << "coef ("<<k<<","<<j<<","<<i<<") = " << (*it) << " ";
        ++it;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

#ifdef __AVEC_XERCES__
void Polynome::Init(DOMNode* n)
{
  DOMNode* sn = n->getFirstChild();
  while(sn)
  {
    if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("degx")))
      mdDegX = atoi(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
    else if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("degy")))
      mdDegY = atoi(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
    else if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("degz")))
    {
      mdDegZ = atoi(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
      mdCoef.resize((mdDegX+1)*(mdDegY+1)*(mdDegZ+1));
      for(std::vector<double>::iterator it = mdCoef.begin();it!=mdCoef.end();++it)(*it)=0.;
    }
    else if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("cst")))
      mdCoef[0] = atof(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
    else
    {
      std::string nom(XMLString::transcode(sn->getNodeName()));
      int dx = 0;
      int dy = 0;
      int dz = 0;
      char tmp[2];
      tmp[1] = '\0';
      for(size_t i=0;(i+1)<nom.size();++i)
      {
        tmp[0]=nom[i+1];
        if (nom[i]=='x')
          dx = atoi(tmp);
        else if (nom[i]=='y')
          dy = atoi(tmp);
        else if (nom[i]=='z')
          dz = atoi(tmp);
      }
      if ((dx>0)||(dy>0)||(dz>0))
        mdCoef[dz*(mdDegX+1)*(mdDegY+1)+dy*(mdDegX+1)+dx] = atof(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
    }
    sn=sn->getNextSibling();
  }
}
#else
void Polynome::Init(cElXMLTree* noeud)
{
  mdDegX = noeud->GetUniqueValInt(std::string("degx"));
  mdDegY = noeud->GetUniqueValInt(std::string("degy"));
  mdDegZ = noeud->GetUniqueValInt(std::string("degz"));
  mdCoef.resize((mdDegX+1)*(mdDegY+1)*(mdDegZ+1));
  std::vector<double>::iterator it = mdCoef.begin();
  for(int i=0;i<=mdDegZ;++i)
  {      
    for(int j=0;j<=mdDegY;++j)
    {
      for(int k=0;k<=mdDegX;++k)
      {
        std::string nom;
        if ((i==0)&&(j==0)&&(k==0))
          nom = std::string("cst");
        else
        {
          std::ostringstream oss;
          bool first = true;
          if (k>0)
          {
            oss << "x" << k ;
            first = false;
          }
          if (j>0)
          {
            if (!first)
              oss << "_";
            oss << "y" << j ;
            first = false;
          }
          if (i>0)
          {
            if (!first)
              oss << "_";
            oss << "z" << i ;
          }
          nom = oss.str();
          
        }
        //std::cout << "nom : "<<nom<<std::endl;
        cElXMLTree * N = NULL;
        std::list<cElXMLTree*> liste = noeud->GetAll(nom);
        if (liste.size()==1)
            N = *(liste.begin());
        if (N != NULL)
        {
            //std::cout << "Non NULL" << std::endl;
          (*it) = N->GetUniqueValDouble();
        }
        else
        {
            //std::cout << "NULL" << std::endl;
          (*it) = 0.;
        }
        ++it;
      }
    }
  }
}
#endif

void Polynome::InitBinary(std::ifstream &fic)
{
	fic.read((char*)&mdDegX,sizeof(mdDegX));
	fic.read((char*)&mdDegY,sizeof(mdDegY));
	fic.read((char*)&mdDegZ,sizeof(mdDegZ));

  	mdCoef.resize((mdDegX+1)*(mdDegY+1)*(mdDegZ+1));
	fic.read((char*)&(mdCoef[0]),(unsigned int) (mdCoef.size()*sizeof(double)));
}

void Polynome::WriteBinary(std::ofstream &fic)const
{
        fic.write((const char*)&mdDegX,sizeof(mdDegX));
        fic.write((const char*)&mdDegY,sizeof(mdDegY));
        fic.write((const char*)&mdDegZ,sizeof(mdDegZ));

        fic.write((const char*)&(mdCoef[0]),(unsigned int) (mdCoef.size()*sizeof(double)));
}

double RTO::Valeur(double x, double y, double z) const
{
  double numVal = Num.Valeur(x,y,z);
  double denVal = Den.Valeur(x,y,z);
  if (std::abs(denVal) < std::numeric_limits<double>::epsilon() ) 
  {
    return std::numeric_limits<double>::infinity();
  }
  return numVal/denVal;
}

void RTO::Afficher()const
{
  std::cout << "Numerateur:"<<std::endl;
  Num.Afficher();
  std::cout << "Denominateur:"<<std::endl;
  Den.Afficher();
}

#ifdef __AVEC_XERCES__
void RTO::Init(DOMNode* n)
{
  DOMNode* sn = n->getFirstChild();
  while(sn)
  {
    if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("polynom3Vreal")))
    {
      DOMNamedNodeMap* att = sn->getAttributes();
      if (!att)
        continue;
      if (!XMLString::compareString(att->getNamedItem(XMLString::transcode("name"))->getNodeValue(),XMLString::transcode("numerateur")))
        Num.Init(sn);
      else if (!XMLString::compareString(att->getNamedItem(XMLString::transcode("name"))->getNodeValue(),XMLString::transcode("denominateur")))
        Den.Init(sn);
    }
    sn=sn->getNextSibling();
  }
}
#else
void RTO::Init(cElXMLTree* noeud)
{
  std::list<cElXMLTree*> noeuds=noeud->GetAll(std::string("polynom3Vreal"));
  std::list<cElXMLTree*>::iterator it,fin=noeuds.end();
  for(it=noeuds.begin();it!=fin;++it)
  {
    std::string name = (*it)->ValAttr("name");
    if (name == std::string("numerateur"))
    {
      Num.Init(*it);
    }
    else if (name == std::string("denominateur"))
    {
      Den.Init(*it);
    }
  }
}
#endif

void RTO::InitBinary(std::ifstream &fic)
{
	Num.InitBinary(fic);
	Den.InitBinary(fic);
}

void RTO::WriteBinary(std::ofstream &fic) const
{
        Num.WriteBinary(fic);
        Den.WriteBinary(fic);
}


void OrientationRTO::ImageAndPx2Obj(double c, double l, const double *z,
                    double &x, double &y)const
{
  double cv=c,lv=l;
  // Si besoin on corrige la distorsion
  if (camera)
  {
     camera->ImageAndPx2Obj(c,l,z,cv,lv);
  }
  // les coord de depart verfiees
  if (cv<MinImageX) cv = MinImageX;
  else if (cv>MaxImageX) cv = MaxImageX;
  if (lv<MinImageY) lv = MinImageY;
  else if (lv>MaxImageY) lv = MaxImageY;
  double X,Y,Z;
  X = (cv-CentreImageX)/CoefImageX;
  Y = (lv-CentreImageY)/CoefImageY;
  Z = (z[0]-CentreImageZ)/CoefImageZ;
  x = ImageX.Valeur(X,Y,Z) * CoefObjetX + CentreObjetX;
  y = ImageY.Valeur(X,Y,Z) * CoefObjetY + CentreObjetY;
}

void OrientationRTO::Objet2ImageInit(double x, double y, const double *z,
                     double &c, double &l)const
{
  // les coord de depart verifiees
  double xv=x,yv=y;
  if (xv<MinObjetX) xv = MinObjetX;
  else if (xv>MaxObjetX) xv = MaxObjetX;
  if (yv<MinObjetY) yv = MinObjetY;
  else if (yv>MaxObjetY) yv = MaxObjetY;
  double X,Y,Z;
  X = (xv-CentreObjetX)/CoefObjetX;
  Y = (yv-CentreObjetY)/CoefObjetY;
  Z = (z[0]-CentreObjetZ)/CoefObjetZ;
  c = ObjetX.Valeur(X,Y,Z) * CoefImageX + CentreImageX;
  l = ObjetY.Valeur(X,Y,Z) * CoefImageY + CentreImageY;
  // Si besoin on applique la correction de distorsion
  if (camera)
  {
     double c2=c,l2=l;
     camera->Objet2ImageInit(c2,l2,z,c,l);
  }
}

double OrientationRTO::GetResolMoyenne() const 
{
  double c=-1.;
  double l=-1.;
  double X,Y,Z;
  X = c/CoefImageX;
  Y = l/CoefImageY;
  Z = 0;
  double x1 = ImageX.Valeur(X,Y,Z) * CoefObjetX + CentreObjetX;
  double y1 = ImageY.Valeur(X,Y,Z) * CoefObjetY + CentreObjetY;
  c=+1.;
  l=+1.;
  X = c/CoefImageX;
  Y = l/CoefImageY;
  Z = 0;
  double x2 = ImageX.Valeur(X,Y,Z) * CoefObjetX + CentreObjetX;
  double y2 = ImageY.Valeur(X,Y,Z) * CoefObjetY + CentreObjetY;

  return std::sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))/(2.*sqrt(2.));
}

bool OrientationRTO::GetPxMoyenne(double * aPxMoy) const 
{
  aPxMoy[0] = CentreImageZ;

  return true;
}

OrientationRTO::OrientationRTO(std::string const &nom):ModuleOrientation(nom)
{
  {
      FILE *  aFP = ElFopen(nom.c_str(),"r");
      if (aFP==0)
      {
         std::cout << "CANNOT OPEN:" << nom << "\n";
	 assert(false);
      }
      ElFclose(aFP);
  }
  MinImageX = 0;
  MinImageY = 0;
  MaxImageX = 0;
  MaxImageY = 0;
  MinObjetX = 0;
  MinObjetY = 0;
  MaxObjetX = 0;
  MaxObjetY = 0;
  camera = NULL;
	bool vXml = true;
	{
		std::ifstream fic(nom.c_str());
		char c;
		fic >> c;
		vXml = (c=='<');
	}
	if (vXml)
		InitXml(nom);
	else
		InitBinary(nom);
}

void OrientationRTO::InitXml(std::string const &nom)
{
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
// Recherche d'un fichier de camera dans le champ sub_code
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
//SYS1
    if (!XMLString::compareString(n->getNodeName(),XMLString::transcode("sys1_center")))
    {
      DOMNode* sn = n->getFirstChild();
      while(sn)
      {
        if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("x")))
          CentreImageX = atof(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
        else if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("y")))
          CentreImageY = atof(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
        else if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("z")))
          CentreImageZ = atof(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
        sn=sn->getNextSibling();
      }
    }
    else if (!XMLString::compareString(n->getNodeName(),XMLString::transcode("sys1_coef")))
    {
      DOMNode* sn = n->getFirstChild();
      while(sn)
      {
        if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("x")))
          CoefImageX = atof(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
        else if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("y")))
          CoefImageY = atof(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
        else if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("z")))
          CoefImageZ = atof(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
        sn=sn->getNextSibling();
      }
    }
    else if (!XMLString::compareString(n->getNodeName(),XMLString::transcode("sys1_min")))
    {
      DOMNode* sn = n->getFirstChild();
      while(sn)
      {
        if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("x")))
          MinImageX = atof(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
        else if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("y")))
          MinImageY = atof(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
        sn=sn->getNextSibling();
      }
    }
    else if (!XMLString::compareString(n->getNodeName(),XMLString::transcode("sys1_max")))
    {
      DOMNode* sn = n->getFirstChild();
      while(sn)
      {
        if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("x")))
          MaxImageX = atof(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
        else if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("y")))
          MaxImageY = atof(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
        sn=sn->getNextSibling();
      }
    }
// SYS2
    if (!XMLString::compareString(n->getNodeName(),XMLString::transcode("sys2_center")))
    {
      DOMNode* sn = n->getFirstChild();
      while(sn)
      {
        if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("x")))
          CentreObjetX = atof(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
        else if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("y")))
          CentreObjetY = atof(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
        else if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("z")))
          CentreObjetZ = atof(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
        sn=sn->getNextSibling();
      }
    }
    else if (!XMLString::compareString(n->getNodeName(),XMLString::transcode("sys2_coef")))
    {
      DOMNode* sn = n->getFirstChild();
      while(sn)
      { 
        if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("x")))
	{
          CoefObjetX = atof(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
	}
        else if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("y")))
          CoefObjetY = atof(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
        else if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("z")))
          CoefObjetZ = atof(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
        sn=sn->getNextSibling();
      }
    }
    else if (!XMLString::compareString(n->getNodeName(),XMLString::transcode("sys2_min")))
    {
      DOMNode* sn = n->getFirstChild();
      while(sn)
      {
        if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("x")))
          MinObjetX = atof(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
        else if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("y")))
          MinObjetY = atof(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
        sn=sn->getNextSibling();
      }
    }
    else if (!XMLString::compareString(n->getNodeName(),XMLString::transcode("sys2_max")))
    {
      DOMNode* sn = n->getFirstChild();
      while(sn)
      {
        if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("x")))
          MaxObjetX = atof(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
        else if (!XMLString::compareString(sn->getNodeName(),XMLString::transcode("y")))
          MaxObjetY = atof(XMLString::transcode(sn->getFirstChild()->getNodeValue()));
        sn=sn->getNextSibling();
      }
    }
    else if (!XMLString::compareString(n->getNodeName(),XMLString::transcode("fct_ratio")))
    {
      DOMNamedNodeMap* att = n->getAttributes();
      if (!att)
        continue;
      if (!XMLString::compareString(att->getNamedItem(XMLString::transcode("name"))->getNodeValue(),XMLString::transcode("x2")))
        ImageX.Init(n);
      else if (!XMLString::compareString(att->getNamedItem(XMLString::transcode("name"))->getNodeValue(),XMLString::transcode("y2")))
        ImageY.Init(n);
      else if (!XMLString::compareString(att->getNamedItem(XMLString::transcode("name"))->getNodeValue(),XMLString::transcode("x1")))
        ObjetX.Init(n);
      else if (!XMLString::compareString(att->getNamedItem(XMLString::transcode("name"))->getNodeValue(),XMLString::transcode("y1")))
        ObjetY.Init(n);
    }
    n = n->getNextSibling();
  }//while
   //Afficher();
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
  {
    cElXMLTree* noeud = tree.GetUnique(std::string("sys1_center"));
    CentreImageX = noeud->GetUniqueValDouble(std::string("x"));
    CentreImageY = noeud->GetUniqueValDouble(std::string("y"));
    CentreImageZ = noeud->GetUniqueValDouble(std::string("z"));
  }
  {
    cElXMLTree* noeud = tree.GetUnique(std::string("sys1_coef"));
    CoefImageX = noeud->GetUniqueValDouble(std::string("x"));
    CoefImageY = noeud->GetUniqueValDouble(std::string("y"));
    CoefImageZ = noeud->GetUniqueValDouble(std::string("z"));
  }
  {
    cElXMLTree* noeud = tree.GetUnique(std::string("sys1_min"));
    MinImageX = noeud->GetUniqueValDouble(std::string("x"));
    MinImageY = noeud->GetUniqueValDouble(std::string("y"));
  }
  {
    cElXMLTree* noeud = tree.GetUnique(std::string("sys1_max"));
    MaxImageX = noeud->GetUniqueValDouble(std::string("x"));
    MaxImageY = noeud->GetUniqueValDouble(std::string("y"));
  }
  {
    cElXMLTree* noeud = tree.GetUnique(std::string("sys2_center"));
    CentreObjetX = noeud->GetUniqueValDouble(std::string("x"));
    CentreObjetY = noeud->GetUniqueValDouble(std::string("y"));
    CentreObjetZ = noeud->GetUniqueValDouble(std::string("z"));
  }
  
  {
    cElXMLTree* noeud = tree.GetUnique(std::string("sys2_coef"));
    CoefObjetX = noeud->GetUniqueValDouble(std::string("x"));
    CoefObjetY = noeud->GetUniqueValDouble(std::string("y"));
    CoefObjetZ = noeud->GetUniqueValDouble(std::string("z"));
  }
  {
    cElXMLTree* noeud = tree.GetUnique(std::string("sys2_min"));
    MinObjetX = noeud->GetUniqueValDouble(std::string("x"));
    MinObjetY = noeud->GetUniqueValDouble(std::string("y"));
  }
  {
    cElXMLTree* noeud = tree.GetUnique(std::string("sys2_max"));
    MaxObjetX = noeud->GetUniqueValDouble(std::string("x"));
    MaxObjetY = noeud->GetUniqueValDouble(std::string("y"));
  }
  
  std::list<cElXMLTree*> noeuds=tree.GetAll(std::string("fct_ratio"));
  std::list<cElXMLTree*>::iterator it,fin=noeuds.end();
  for(it=noeuds.begin();it!=fin;++it)
  {
    std::string name = (*it)->ValAttr("name");
    if (name == std::string("x2")) // RTO donnant x en Objet en fonction de col,lig,Z
    {
      ImageX.Init(*it);
    }
    else if (name == std::string("y2")) // RTO donnant y en Objet en fonction de col,lig,Z
    {
      ImageY.Init(*it);
    }
    else if (name == std::string("x1")) // RTO donnant col (image) en fonction de x,y,Z
    {
      ObjetX.Init(*it);
    }
    else if (name == std::string("y1")) // RTO donnant lig (image) en fonction de x,y,Z
    {
      ObjetY.Init(*it);
    }
  }
  //Afficher();
#endif
 if ((nomCamera.length()>0)&&(nomCamera!=std::string("*")))
        {
                std::cout << "Chargement de la camera : "<<nomCamera<<std::endl;
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
}
void OrientationRTO::InitBinary(std::string const &nom)
{
	std::ifstream fic(nom.c_str(),ios::binary);
	fic.read((char*)&CentreImageX,sizeof(CentreImageX));
	fic.read((char*)&CentreImageY,sizeof(CentreImageY));
	fic.read((char*)&CentreImageZ,sizeof(CentreImageZ));
	fic.read((char*)&CoefImageX,sizeof(CoefImageX));
        fic.read((char*)&CoefImageY,sizeof(CoefImageY));
        fic.read((char*)&CoefImageZ,sizeof(CoefImageZ));
        fic.read((char*)&MinImageX,sizeof(MinImageX));
        fic.read((char*)&MinImageY,sizeof(MinImageY));
        fic.read((char*)&MaxImageX,sizeof(MaxImageX));
        fic.read((char*)&MaxImageY,sizeof(MaxImageY));
	fic.read((char*)&CentreObjetX,sizeof(CentreObjetX));
        fic.read((char*)&CentreObjetY,sizeof(CentreObjetY));
        fic.read((char*)&CentreObjetZ,sizeof(CentreObjetZ));
        fic.read((char*)&CoefObjetX,sizeof(CoefObjetX));
        fic.read((char*)&CoefObjetY,sizeof(CoefObjetY));
        fic.read((char*)&CoefObjetZ,sizeof(CoefObjetZ));
        fic.read((char*)&MinObjetX,sizeof(MinObjetX));
        fic.read((char*)&MinObjetY,sizeof(MinObjetY));
        fic.read((char*)&MaxObjetX,sizeof(MaxObjetX));
        fic.read((char*)&MaxObjetY,sizeof(MaxObjetY));
	ImageX.InitBinary(fic);
	ImageY.InitBinary(fic);
	ObjetX.InitBinary(fic);
	ObjetY.InitBinary(fic);
	fic.close();
}

void OrientationRTO::WriteBinary(std::string const &nom)const
{
        std::ofstream fic(nom.c_str(),ios::binary);
        fic.write((const char*)&CentreImageX,sizeof(CentreImageX));
        fic.write((const char*)&CentreImageY,sizeof(CentreImageY));
        fic.write((const char*)&CentreImageZ,sizeof(CentreImageZ));
        fic.write((const char*)&CoefImageX,sizeof(CoefImageX));
        fic.write((const char*)&CoefImageY,sizeof(CoefImageY));
        fic.write((const char*)&CoefImageZ,sizeof(CoefImageZ));
        fic.write((const char*)&MinImageX,sizeof(MinImageX));
        fic.write((const char*)&MinImageY,sizeof(MinImageY));
        fic.write((const char*)&MaxImageX,sizeof(MaxImageX));
        fic.write((const char*)&MaxImageY,sizeof(MaxImageY));
        fic.write((const char*)&CentreObjetX,sizeof(CentreObjetX));
        fic.write((const char*)&CentreObjetY,sizeof(CentreObjetY));
        fic.write((const char*)&CentreObjetZ,sizeof(CentreObjetZ));
        fic.write((const char*)&CoefObjetX,sizeof(CoefObjetX));
        fic.write((const char*)&CoefObjetY,sizeof(CoefObjetY));
        fic.write((const char*)&CoefObjetZ,sizeof(CoefObjetZ));
        fic.write((const char*)&MinObjetX,sizeof(MinObjetX));
        fic.write((const char*)&MinObjetY,sizeof(MinObjetY));
        fic.write((const char*)&MaxObjetX,sizeof(MaxObjetX));
        fic.write((const char*)&MaxObjetY,sizeof(MaxObjetY));
        ImageX.WriteBinary(fic);
        ImageY.WriteBinary(fic);
        ObjetX.WriteBinary(fic);
        ObjetY.WriteBinary(fic);
        fic.close();
}

void OrientationRTO::Afficher() const
{
  std::cout << "CentreImage : "<<CentreImageX << " "<<CentreImageY << " "<<CentreImageZ << std::endl;
  std::cout << "CoefImage : "<<CoefImageX << " " <<CoefImageY<< " "<<CoefImageZ << std::endl;
  std::cout << "MinImage : "<<MinImageX<<" "<<MinImageY<<std::endl;
  std::cout << "MaxImage : "<<MaxImageX<<" "<<MaxImageY<<std::endl;
  std::cout << "RTO Image -> Objet X : " <<std::endl;
  ImageX.Afficher();
  std::cout << "RTO Image -> Objet Y : " <<std::endl;
  ImageY.Afficher();
  
  std::cout << "CentreObjet : "<<CentreObjetX << " "<<CentreObjetY << " "<<CentreObjetZ << std::endl;
  std::cout << "CoefObjet : "<<CoefObjetX << " " <<CoefObjetY<< " "<<CoefObjetZ << std::endl;
  std::cout << "MinObjet : "<<MinObjetX<<" "<<MinObjetY<<std::endl;
  std::cout << "MaxObjet : "<<MaxObjetX<<" "<<MaxObjetY<<std::endl;
  std::cout << "RTO Objet -> Image X : " <<std::endl;
  ObjetX.Afficher();
  std::cout << "RTO Objet -> Image Y : " <<std::endl;
  ObjetY.Afficher();
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
