#include "readwrite.h"


using namespace std;

QString qtError (const QXmlStreamReader& xmlReader,  const QString& s) {
	if (xmlReader.error() == QXmlStreamReader::CustomError)
		return (s + conv(QObject::tr(" Use error.")));
	else if (xmlReader.error() == QXmlStreamReader::NotWellFormedError)
		return (s + conv(QObject::tr(" problem in XML file structure.")));
	else if (xmlReader.error() == QXmlStreamReader::PrematureEndOfDocumentError)
		return (s + conv(QObject::tr(" Uncomplete XML file.")));
	else if (xmlReader.error() == QXmlStreamReader::UnexpectedElementError)
		return (s + conv(QObject::tr(" Unexpected element.")));
	else
		return QString();
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

XmlTree::XmlTree() : fichier(QString()), mainTag(XmlTag()), otherTags(QList<XmlTag>()) {}
XmlTree::XmlTree(const XmlTree& xmlTree) { copie(xmlTree); }
XmlTree::XmlTree(const QString& f) : fichier(f), mainTag(XmlTag()), otherTags(QList<XmlTag>()) {}
XmlTree::XmlTree(const QString& f, const XmlTag& m, const QList<XmlTag>& o) : fichier(f), mainTag(m), otherTags(o) {}
XmlTree::~XmlTree() {}

XmlTree& XmlTree::operator=(const XmlTree& xmlTree) {
	if (this!=&xmlTree) copie(xmlTree);
	return *this;
}

void XmlTree::copie(const XmlTree& xmlTree) {
	fichier = xmlTree.getFichier();
	mainTag = xmlTree.getMainTag();
	otherTags = xmlTree.getOtherTags();
}

const QString& XmlTree::getFichier() const { return fichier; }
const XmlTag& XmlTree::getMainTag() const { return mainTag; }
XmlTag& XmlTree::modifMainTag() { return mainTag; }
const QList<XmlTag>& XmlTree::getOtherTags() const { return otherTags; }

void XmlTree::setFichier(const QString& f) { fichier = f; }
void XmlTree::setMainTag(const XmlTag& m) { mainTag = m; }
void XmlTree::setOtherTags(const QList<XmlTag>& o) { otherTags = o; }

QString XmlTree::lire(bool all) {	
	QFile file(fichier);
	if (!file.open(QFile::ReadOnly | QFile::Text)) {
		return conv(QObject::tr("Fail to open file %1")).arg(fichier);
	}
	QXmlStreamReader xmlReader(&file);
	XmlTag* currentTag = 0;

	while (!xmlReader.atEnd()) {
		while (!xmlReader.atEnd() && !xmlReader.isStartElement() && !xmlReader.isEndElement())
			xmlReader.readNext();
		if (xmlReader.isStartElement()) {
			//nom et parent
			if (currentTag==0 && mainTag.getNom().isEmpty()) {
				mainTag.setNom(xmlReader.name().toString());
				currentTag = &mainTag;
			} else if (currentTag==0) {
				otherTags.push_back( XmlTag(xmlReader.name().toString(), 0) );
				currentTag = &(otherTags.last());
			} else {
				currentTag->addEnfant( XmlTag(xmlReader.name().toString(), currentTag) );
				currentTag = &(currentTag->modifEnfants().last());
			}
			//attributs
			if (xmlReader.attributes().count()>0) {
				for (int i=0; i<xmlReader.attributes().count(); i++) {
					currentTag->addAttribut(xmlReader.attributes().at(i).name().toString(),xmlReader.attributes().at(i).value().toString());
				}
			}
			//contenu
			xmlReader.readNext();
			if (xmlReader.tokenType()==QXmlStreamReader::Characters) currentTag->setContenu(xmlReader.text().toString().trimmed().simplified());
		} else if (xmlReader.isEndElement()) {
			if (currentTag==0) return conv(QObject::tr("XML file structure error ; unexpected closing element met.")).arg(fichier);
			currentTag = currentTag->getParent();
			xmlReader.readNext();
		} else
			break;
	}
	while (!xmlReader.atEnd())
		xmlReader.readNext();

	if (all) return qtError(xmlReader,fichier);
	else return QString();
}

bool XmlTree::ecrire() {
	QFile oldFile(fichier);
	if (oldFile.exists()) oldFile.remove();
	QFile file(fichier);
	if (!file.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) return false;
	QXmlStreamWriter xmlWriter(&file);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	QList<XmlTag> l = otherTags;
	l.push_front(mainTag);
	ecrireListe(xmlWriter, &l);

	xmlWriter.writeEndDocument();
	return true;
}

void XmlTree::ecrireListe(QXmlStreamWriter& xmlWriter, const QList<XmlTag>* currentList) {
	for (QList<XmlTag>::const_iterator itTag=currentList->begin(); itTag!=currentList->end(); itTag++) {
		if (itTag->getEnfants().count()>0) { 
			xmlWriter.writeStartElement(itTag->getNom());
			if (itTag->getAttributs().count()>0) {
				for (QList<std::pair<QString,QString> >::const_iterator it=itTag->getAttributs().begin(); it!=itTag->getAttributs().end(); it++)
					xmlWriter.writeAttribute(it->first,it->second);
			}
			ecrireListe(xmlWriter, &(itTag->getEnfants()));
			xmlWriter.writeEndElement();
		} else
			xmlWriter.writeTextElement(itTag->getNom(), itTag->getContenu());
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

XmlTag::XmlTag() : nom(QString()), contenu(QString()), parent(0), enfants(QList<XmlTag>()), attributs(QList<pair<QString,QString> >()) {}
XmlTag::XmlTag(const XmlTag& xmlTag) { copie(xmlTag); }
XmlTag::XmlTag(const QString& n, XmlTag* const p) : nom(n), contenu(QString()), parent(p), enfants(QList<XmlTag>()), attributs(QList<pair<QString,QString> >()) {}
XmlTag::XmlTag(const QString& n, const QString& c, XmlTag* const p) : nom(n), contenu(c), parent(p), enfants(QList<XmlTag>()), attributs(QList<pair<QString,QString> >()) {}
XmlTag::XmlTag(const QString& n, const QString& c, XmlTag* const p, const QList<XmlTag>& e, const QList<pair<QString,QString> >& a) : nom(n), contenu(c), parent(p), enfants(e), attributs(a) {}
XmlTag::~XmlTag() {}

XmlTag& XmlTag::operator=(const XmlTag& xmlTag) {
	if (this!=&xmlTag) copie(xmlTag);
	return *this;
}

void XmlTag::copie(const XmlTag& xmlTag) {
	nom = xmlTag.getNom();
	contenu = xmlTag.getContenu();
	parent = xmlTag.getParent();
	enfants = xmlTag.getEnfants();
	attributs = xmlTag.getAttributs();
}

const QString& XmlTag::getNom() const { return nom; }
const QString& XmlTag::getContenu() const { return contenu; }
XmlTag* const XmlTag::getParent() const { return parent; }
const QList<XmlTag>& XmlTag::getEnfants() const { return enfants; }
QList<XmlTag>& XmlTag::modifEnfants() { return enfants; }
const QList<pair<QString,QString> >& XmlTag::getAttributs() const { return attributs; }
QList<pair<QString,QString> >& XmlTag::modifAttributs() { return attributs; }

void XmlTag::setNom(const QString& n) { nom = n; }
void XmlTag::setContenu(const QString& c) { contenu = c; }
void XmlTag::setParent(XmlTag* const p) { parent = p; }
void XmlTag::setEnfants(const QList<XmlTag>& e) { enfants = e; }
void XmlTag::setAttributs(const QList<pair<QString,QString> >& a) { attributs = a; }

void XmlTag::addEnfant(const XmlTag& child) {
	enfants.push_back(child);
}
void XmlTag::addAttribut(const QString& attNom, const QString& attValue) {
	attributs.push_back( pair<QString,QString>(attNom,attValue) );
}

const XmlTag& XmlTag::getEnfant(const QString& childName, bool* ok) const {
	if (ok!=0) *ok = true;
	for (QList<XmlTag>::const_iterator it=enfants.begin(); it!=enfants.end(); it++) {
		if (it->nom==childName) return *it;
	}
	if (ok!=0) *ok = false;
	return *this;
}
const XmlTag& XmlTag::getEnfant(const char* childName, bool* ok) const { return getEnfant(QString(childName),ok); }
const QList<const XmlTag*> XmlTag::getEnfants(const QString& childName, bool* ok) const {
	QList<const XmlTag*> l;
	for (QList<XmlTag>::const_iterator it=enfants.begin(); it!=enfants.end(); it++) {
		if (it->nom==childName) l.push_back(&(*it));
	}
	if (ok!=0) *ok = (l.count()>0);
	return l;
}
const QList<const XmlTag*> XmlTag::getEnfants(const char* childName, bool* ok) const { return getEnfants(QString(childName),ok); }

const QString& XmlTag::getAttribut(const QString& attName, bool* ok) const {
	if (ok!=0) *ok = true;
	for (QList<pair<QString,QString> >::const_iterator it=attributs.begin(); it!=attributs.end(); it++) {
		if (it->first==attName) return it->second;
	}
	if (ok!=0) *ok = false;
	return nom;
}
const QString& XmlTag::getAttribut(const char* attName, bool* ok) const { return getAttribut(QString(attName),ok); }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx//
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//enregistrement du calcul en cours

bool ParamCalcul::ecrire(const ParamMain& paramMain, const QString& fichier) {
//écriture des différents paramètres de l'interface et des fichiers utilisés dans les calculs
	QFile oldFile(fichier);
	if (oldFile.exists()) {
		oldFile.remove();
	}
	QFile file(fichier);
	if (!file.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) {
			return false;
	}
	QXmlStreamWriter xmlWriter(&file);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	xmlWriter.writeStartElement(QString("Calcul"));

	ParamMain::Mode mode = paramMain.getCurrentMode();
	QVector<pair<ParamMain::Mode,QString> >::const_iterator it=paramMain.getTradMode().begin();
	while (it->first != mode && it!=paramMain.getTradMode().end()) {
		it++;
	}
	xmlWriter.writeTextElement(QString("Etape"), it->second);

	if (paramMain.getCurrentMode()!=ParamMain::BeginMode) {
		QString repertoire = fichier.section("/",0,-2);
		xmlWriter.writeTextElement(QString("Repertoire"), QDir(repertoire).relativeFilePath(paramMain.getDossier()));

		if (paramMain.getCurrentMode()!=ParamMain::ImageMode) {
			//params chantier et Pastis
			xmlWriter.writeStartElement(QString("Chantier"));

				ParamPastis::TypeChantier typChan = paramMain.getParamPastis().getTypeChantier();
				QVector<pair<ParamPastis::TypeChantier,QString> >::const_iterator it=paramMain.getParamPastis().getTradTypChan().begin();
				while (it->first != typChan && it!=paramMain.getParamPastis().getTradTypChan().end()) {
					it++;
				}
				xmlWriter.writeTextElement(QString("Type"), it->second);

				xmlWriter.writeTextElement(QString("Sz"), QVariant(paramMain.getParamPastis().getLargeurMax()).toString());
				if (paramMain.getParamPastis().getMultiScale()) {
					xmlWriter.writeTextElement(QString("MultiEchelle"), QString("true"));
					xmlWriter.writeTextElement(QString("NbPtMin"), QVariant(paramMain.getParamPastis().getNbPtMin()).toString());
					xmlWriter.writeTextElement(QString("Sz2"), QVariant(paramMain.getParamPastis().getLargeurMax2()).toString());
				}		
			xmlWriter.writeEndElement();	//Chantier
		
			if (paramMain.getCurrentMode()==ParamMain::PointsEnCours) {
				xmlWriter.writeTextElement(QString("Avancement"), QVariant(paramMain.getAvancement()).toString());

			} else if (paramMain.getCurrentMode()!=ParamMain::PointsMode) {
				xmlWriter.writeStartElement(QString("Orientation"));
					//auto-calibration
					if (paramMain.getParamApero().getAutoCalib().count()>0) {
						QString t;
						for (int i=0; i<paramMain.getParamApero().getAutoCalib().count(); i++)
							t = t + paramMain.getParamApero().getAutoCalib().at(i) + QString(" ");
						xmlWriter.writeTextElement(QString("AutoCalib"), t);
					}
					//filtrage
					if (paramMain.getParamApero().getFiltrage()) xmlWriter.writeTextElement(QString("Filtrage"), QString("true"));
					if (!paramMain.getParamApero().getCalcPts3D()) xmlWriter.writeTextElement(QString("PtsHomol3D"), QString("false"));	//calculés par défaut
					//multi-échelle
					xmlWriter.writeTextElement(QString("MultiEchelle"), QVariant(paramMain.getParamApero().getMultiechelle()).toString());
					if (paramMain.getParamApero().getMultiechelle()) {
						QString text;
						for (int i=0; i<paramMain.getParamApero().getCalibFigees().count(); i++)
							text += QVariant(paramMain.getParamApero().getCalibFigees().at(i)).toString() + QString(" ");
						xmlWriter.writeTextElement(QString("PosesFigees"), text);
					}
					//orientation initiale
					if (paramMain.getParamApero().getUseOriInit())
						xmlWriter.writeTextElement(QString("OriInit"), paramMain.getParamApero().getDirOriInit());
					//orientation absolue
					if (paramMain.getParamApero().getUserOrientation().getOrientMethode()==1) {	//orientation utilisateur
						xmlWriter.writeTextElement(QString("OrientMethode"), QString("Manuelle"));
						if (paramMain.getParamApero().getUserOrientation().getBascOnPlan()) {
							//plan
							xmlWriter.writeTextElement(QString("ImgMasquePlan"), paramMain.getParamApero().getUserOrientation().getImgMasque());
							//direction
							xmlWriter.writeTextElement(QString("Image1"), paramMain.getParamApero().getUserOrientation().getImage1());
							QPoint P1 = paramMain.getParamApero().getUserOrientation().getPoint1();
							xmlWriter.writeTextElement(QString("Point1"), QVariant(P1.x()).toString()+QString(" ")+QVariant(P1.y()).toString());
							xmlWriter.writeTextElement(QString("Image2"), paramMain.getParamApero().getUserOrientation().getImage2());
							QPoint P2 = paramMain.getParamApero().getUserOrientation().getPoint2();
							xmlWriter.writeTextElement(QString("Point2"), QVariant(P2.x()).toString()+QString(" ")+QVariant(P2.y()).toString());
						}
					} else if (paramMain.getParamApero().getUserOrientation().getOrientMethode()==2) {	//orientation absolue d'une image
						xmlWriter.writeTextElement(QString("OrientMethode"), QString("ImageGeoref"));
						xmlWriter.writeTextElement(QString("Image"), paramMain.getParamApero().getUserOrientation().getImageGeoref());
						if (!paramMain.getParamApero().getUserOrientation().getGeorefFile().isEmpty())
							xmlWriter.writeTextElement(QString("ByFile"), paramMain.getParamApero().getUserOrientation().getGeorefFile());
						else {
							//centre
							QString l1;
							for (int i=0; i<3; i++)
								l1 += paramMain.getParamApero().getUserOrientation().getCentreAbs().at(i)+QString(" ");
							xmlWriter.writeTextElement(QString("Centre"), l1);
							//rotation
							xmlWriter.writeStartElement(QString("Rotation"));
							for (int k=0; k<3; k++) {
								QString l2;
								for (int i=0; i<3; i++)
									l2 += paramMain.getParamApero().getUserOrientation().getRotationAbs().at(i+3*k)+ (QString(" "));
								xmlWriter.writeTextElement(QString("L%1").arg(k+1), l2);
							}
							xmlWriter.writeEndElement();	//Rotation
						}
					}
					if ((paramMain.getParamApero().getUserOrientation().getOrientMethode()==1 && paramMain.getParamApero().getUserOrientation().getFixEchelle())
					|| paramMain.getParamApero().getUserOrientation().getOrientMethode()==2) {
						//échelle
						QString l1, l2;
						for (int i=0; i<4; i++) {
							l1 += paramMain.getParamApero().getUserOrientation().getImages().at(i)+QString(" ");
							QPoint P = paramMain.getParamApero().getUserOrientation().getPoints().at(i);
							l2 += QVariant(P.x()).toString()+QString(" ")+QVariant(P.y()).toString()+QString(" ");
						}
						xmlWriter.writeTextElement(QString("ImagesEchelle"), l1);
						xmlWriter.writeTextElement(QString("PointsEchelle"), l2);
						xmlWriter.writeTextElement(QString("Distance"), QVariant(paramMain.getParamApero().getUserOrientation().getDistance()).toString());
					}
					if (paramMain.getParamApero().getUserOrientation().getOrientMethode()==3) {	//géoréférencement par points GPS
						xmlWriter.writeTextElement(QString("OrientMethode"), QString("Appuis"));
						xmlWriter.writeTextElement(QString("PointsGPS"), paramMain.getParamApero().getUserOrientation().getPointsGPS());
						xmlWriter.writeTextElement(QString("AppuisImage"), paramMain.getParamApero().getUserOrientation().getAppuisImg());
					} else if (paramMain.getParamApero().getUserOrientation().getOrientMethode()==4) {	//sommets GPS
						xmlWriter.writeTextElement(QString("OrientMethode"), QString("Sommets"));
						xmlWriter.writeTextElement(QString("SommetsGPS"), paramMain.getParamApero().getUserOrientation().getPointsGPS());
					}
				xmlWriter.writeEndElement();	//Orientation

				if (paramMain.getCurrentMode()==ParamMain::PoseEnCours) {
					xmlWriter.writeTextElement(QString("Avancement"), QVariant(paramMain.getAvancement()).toString());

				} else if (paramMain.getCurrentMode()!=ParamMain::PoseMode) {

					//params MicMac
					xmlWriter.writeStartElement(QString("Cartes_Profondeur"));
						for (int i=0; i<paramMain.getParamMicmac().count(); i++) {
							xmlWriter.writeStartElement(QString("Carte"));
								if (paramMain.getCurrentMode()==ParamMain::CarteEnCours
								&& !paramMain.getParamMicmac().at(i).getACalculer())
									xmlWriter.writeTextElement(QString("Recalculer"), QString("non"));
								xmlWriter.writeTextElement(QString("RefImg"), paramMain.getParamMicmac().at(i).getImageDeReference());

								QString s = *(paramMain.getParamMicmac().at(i).getImagesCorrel().begin());
								for (QStringList::const_iterator it=paramMain.getParamMicmac().at(i).getImagesCorrel().begin()+1; it!=paramMain.getParamMicmac().at(i).getImagesCorrel().end(); it++)
									s += QString(" ") + *it;
								xmlWriter.writeTextElement(QString("ImgsCorrel"), s);

								xmlWriter.writeTextElement(QString("Repere"), (paramMain.getParamMicmac().at(i).getRepere())? QString("Image") : (paramMain.getParamMicmac().at(i).getAutreRepere())? QString("Special") : QString("Terrain"));
								if (!(paramMain.getParamMicmac().at(i).getRepere()) && paramMain.getParamMicmac().at(i).getDoOrtho()) {
									xmlWriter.writeTextElement(QString("DoOrtho"), QString("Oui"));
									QString s = *(paramMain.getParamMicmac().at(i).getImgsOrtho().begin());
									for (QStringList::const_iterator it=paramMain.getParamMicmac().at(i).getImgsOrtho().begin()+1; it!=paramMain.getParamMicmac().at(i).getImgsOrtho().end(); it++)
										s += QString(" ") + *it;
									xmlWriter.writeTextElement(QString("ImgsOrtho"), s);
									xmlWriter.writeTextElement(QString("EchelleOrtho"), QVariant(paramMain.getParamMicmac().at(i).getEchelleOrtho()).toString());
								}
								xmlWriter.writeTextElement(QString("Intervalle"), QString("%1 %2").arg(paramMain.getParamMicmac().at(i).getInterv().first).arg(paramMain.getParamMicmac().at(i).getInterv().second));
								xmlWriter.writeTextElement(QString("Regul"), (!paramMain.getParamMicmac().at(i).getDiscontinuites())? QString()
														: QString("%1 %2").arg(paramMain.getParamMicmac().at(i).getSeuilZ()).arg(paramMain.getParamMicmac().at(i).getSeuilZRelatif()));
							xmlWriter.writeEndElement();	//Carte
						}
					xmlWriter.writeEndElement();	//Cartes_Profondeur

					if (paramMain.getCurrentMode()==ParamMain::CarteEnCours) {
						xmlWriter.writeTextElement(QString("Avancement"), QVariant(paramMain.getAvancement()).toString());

					}
				}
			}
		}

		//écriture des images et des fichiers de calibration
		const QVector<ParamImage>* correspImgCalib = &(paramMain.getCorrespImgCalib());
		xmlWriter.writeStartElement(QString("Images"));
		for (int i=0; i<correspImgCalib->size(); i++) {
			xmlWriter.writeStartElement(QString("Image"));
				xmlWriter.writeTextElement(QString("ImageRAW"), correspImgCalib->at(i).getImageRAW());
				if (paramMain.getCurrentMode()!=ParamMain::ImageMode) {
					if (paramMain.getCurrentMode()!=ParamMain::PointsEnCours || paramMain.getAvancement()>=PastisThread::Conversion)
						xmlWriter.writeTextElement(QString("ImageTif"), correspImgCalib->at(i).getImageTif());
					if (paramMain.getCurrentMode()!=ParamMain::PointsEnCours 
					&& !paramMain.getParamApero().getImgToOri().contains(correspImgCalib->at(i).getImageTif())) {
						xmlWriter.writeTextElement(QString("Orientable"), QString("non"));
					}
				}
			xmlWriter.writeEndElement();	//Image
		}
		xmlWriter.writeEndElement();	//Images
		if (paramMain.getCurrentMode()!=ParamMain::ImageMode) {
			xmlWriter.writeStartElement(QString("Calibrations"));
			for (int i=0; i<paramMain.getParamPastis().getCalibFiles().count(); i++) {
				xmlWriter.writeStartElement(QString("Calibration"));
					xmlWriter.writeTextElement(QString("Fichier"), paramMain.getParamPastis().getCalibFiles().at(i).first);
					xmlWriter.writeTextElement(QString("Focale"), QVariant(paramMain.getParamPastis().getCalibFiles().at(i).second).toString());
				xmlWriter.writeEndElement();	//Calibration
			}	
			xmlWriter.writeEndElement();	//Calibrations
		}
	}
	xmlWriter.writeEndElement();	//Calcul

	xmlWriter.writeEndDocument();
	return true;
}

QString ParamCalcul::lire(ParamMain& paramMain, const QString& fichier) {
	QString sbase = conv(QObject::tr("Backup file %1 reading :\n")).arg(fichier);
	paramMain.setCurrentMode(ParamMain::BeginMode);

	XmlTree xmlTree(fichier);
	QString err = xmlTree.lire();
	if (!err.isEmpty()) return err;

	if (xmlTree.getMainTag().getNom()!=QString("Calcul")) return sbase+conv(QObject::tr("This file is not a computation file.")); 
	bool ok;
	XmlTag tag0;
	XmlTag& tag = *(&tag0);

	//étape
	tag = xmlTree.getMainTag().getEnfant("Etape",&ok);
	if (!ok) return sbase+conv(QObject::tr("Step not found.")); 
	QString text = tag.getContenu();
	int idx=0;
	while (idx<paramMain.getTradMode().count()) {
		if (paramMain.getTradMode().at(idx).second == text) break;
		idx++;		
	}
	if (idx==paramMain.getTradMode().count())
		return sbase+conv(QObject::tr("Fail to recognize computation step."));
	paramMain.setCurrentMode(paramMain.getTradMode().at(idx).first);

	if (paramMain.getCurrentMode()!=ParamMain::BeginMode) {   //ImageMode, PointsEnCours, PointsMode, PoseEnCours, PoseMode, CarteEnCours, EndMode
		//répertoire des images
		QString dir = fichier.section("/",0,-2);
		tag = xmlTree.getMainTag().getEnfant("Repertoire",&ok);
		if (!ok) return sbase+conv(QObject::tr("Data directory not found."));
		QString repertoire = tag.getContenu();
		
		#if ELISE_windows
			if ( repertoire[1]!=QChar(':') )
		#else
			if ( repertoire.left(1)!=QString("/") )
		#endif
				repertoire = QDir(dir+QString("/")+repertoire).canonicalPath();

		repertoire = QDir(repertoire).absolutePath()+QString("/");
		paramMain.setDossier(repertoire);
		cout << QObject::tr("Data directory : ").toStdString(); cout << repertoire.toStdString() << endl;
	
		if (paramMain.getCurrentMode()!=ParamMain::ImageMode) {   //PointsEnCours, PointsMode, PoseEnCours, PoseMode, CarteEnCours, EndMode
			//chantier
			const XmlTag* chantier = &(xmlTree.getMainTag().getEnfant("Chantier",&ok));
			if (!ok) return sbase+conv(QObject::tr("Survey parameters not found."));

			tag = chantier->getEnfant("Type",&ok);
			if (!ok) return sbase+conv(QObject::tr("Survey type not found."));
			QString typChan = tag.getContenu();
			QVector<pair<ParamPastis::TypeChantier,QString> >::const_iterator it=paramMain.getParamPastis().getTradTypChan().begin();
			while (it->second != typChan && it!=paramMain.getParamPastis().getTradTypChan().end()) it++;
			if (it==paramMain.getParamPastis().getTradTypChan().end()) return sbase+conv(QObject::tr("Fail to recognize survey type."));
			paramMain.modifParamPastis().setTypeChantier(it->first);

			tag = chantier->getEnfant("Sz",&ok);
			if (!ok) return sbase+conv(QObject::tr("Rescaled image size not found."));		
			int l = tag.getContenu().toInt(&ok);
			if (!ok || l==0) 
				return sbase+conv(QObject::tr("Uncorrect image width for rescaling."));
			paramMain.modifParamPastis().setLargeurMax(l);

			tag = chantier->getEnfant("MultiEchelle",&ok);
			if (ok) paramMain.modifParamPastis().setMultiScale(tag.getContenu()==QString("true"));
			else paramMain.modifParamPastis().setMultiScale(false);

			if (paramMain.getParamPastis().getMultiScale()) {
				tag = chantier->getEnfant("NbPtMin",&ok);
				if (!ok) return sbase+conv(QObject::tr("Tie-point threshold for computation second step not found."));
				paramMain.modifParamPastis().setNbPtMin(tag.getContenu().toInt(&ok));
				if (!ok) return sbase+conv(QObject::tr("Tie-point threshold for computation second step is unvalid."));

				tag = chantier->getEnfant("Sz2",&ok);
				if (!ok) return sbase+conv(QObject::tr("Rescaled image size for tie-point computation second step not found."));
				paramMain.modifParamPastis().setLargeurMax2(tag.getContenu().toInt(&ok));
				if (!ok) return sbase+conv(QObject::tr("Rescaled image size for tie-point computation second step is unvalid."));
			}

			if (paramMain.getCurrentMode()==ParamMain::PointsEnCours) {   //PointsEnCours
				tag = xmlTree.getMainTag().getEnfant("Avancement",&ok);			
				if (!ok) return sbase+conv(QObject::tr("Computation step not found."));		
				int avancement = tag.getContenu().toInt(&ok);
				if (!ok) return sbase+conv(QObject::tr("Uncorrect computation step ."));
				paramMain.setAvancement(avancement);

				if (paramMain.getAvancement()==PastisThread::Conversion || paramMain.getAvancement()==PastisThread::FiltrCpls || paramMain.getAvancement()==PastisThread::PtsInteret) {
					QString ps;
					if (avancement==PastisThread::FiltrCpls) ps = QString("b");
					else if (avancement==PastisThread::PtsInteret) ps = QString("c");					
					paramMain.setMakeFile(QString("MK")+paramMain.getDossier().section("/",-2,-2)+ps);
				}
			}

			else if (paramMain.getCurrentMode()!=ParamMain::PointsMode) {   //PoseEnCours, PoseMode, CarteEnCours, EndMode
				//orientations
				const XmlTag* orientation = &(xmlTree.getMainTag().getEnfant("Orientation",&ok));
				if (!ok) return sbase+conv(QObject::tr("Orientation parameters not found."));

				tag = orientation->getEnfant("AutoCalib",&ok);
				if (ok) {
					paramMain.modifParamApero().modifAutoCalib().clear();
					int i=0;
					text = tag.getContenu();
					while (!text.section(" ",i,i).isEmpty()) {
						paramMain.modifParamApero().modifAutoCalib().push_back(text.section(" ",i,i));
						i++;
					}
				}

				tag = orientation->getEnfant("Filtrage",&ok);
				if (ok) paramMain.modifParamApero().setFiltrage(tag.getContenu()==QString("true"));

				tag = orientation->getEnfant("PtsHomol3D",&ok);
				if (ok) paramMain.modifParamApero().setFiltrage(tag.getContenu()!=QString("false"));

				tag = orientation->getEnfant("MultiEchelle",&ok);
				if (!ok) return sbase+conv(QObject::tr("Multiscaling parameter not found."));
				paramMain.modifParamApero().setMultiechelle(tag.getContenu()==QString("true"));

				tag = orientation->getEnfant("OriInit",&ok);
				if (ok) {
					paramMain.modifParamApero().setUseOriInit(true);
					paramMain.modifParamApero().setDirOriInit(tag.getContenu());
				}
		
				if (paramMain.getParamApero().getMultiechelle()) {
					tag = orientation->getEnfant("PosesFigees",&ok);
					if (!ok) return sbase+conv(QObject::tr("Fixed orientations not found."));	
					paramMain.modifParamApero().modifCalibFigees().clear();
					int i=0;
					text = tag.getContenu();
					while (!text.section(" ",i,i).isEmpty()) {
						bool ok = false;
						int calib = text.section(" ",i,i).toInt(&ok);
						if (!ok || calib==0) return sbase+conv(QObject::tr("Fail to recognize a fixed orientation."));	
						paramMain.modifParamApero().modifCalibFigees().push_back(calib);
						i++;
					}
					if (paramMain.getParamApero().getCalibFigees().count()==0)
						return sbase+conv(QObject::tr("No fixed orientations found."));	
				}
					
				tag = orientation->getEnfant("OrientMethode",&ok);
				if (ok) {
					QString orientMethode = tag.getContenu();
					paramMain.modifParamApero().modifUserOrientation() = UserOrientation();

					//orientation manuelle
					if (orientMethode==QString("Manuelle")) {
						paramMain.modifParamApero().modifUserOrientation().setOrientMethode(1);

						tag = orientation->getEnfant("ImgMasquePlan",&ok);
						bool ok2;
						orientation->getEnfant("ImagesEchelle",&ok2);
						if (!ok && !ok2) return sbase+conv(QObject::tr("No parameters found for user orientation (either an horizontal plan or images for rescaling)."));	//plan+dir ou échelle ou les 2
						//plan
						if (ok) {
							paramMain.modifParamApero().modifUserOrientation().setBascOnPlan(true);
							paramMain.modifParamApero().modifUserOrientation().setImgMasque(tag.getContenu());

							//direction
							tag = orientation->getEnfant("Image1",&ok);
							if (!ok) return sbase+conv(QObject::tr("First image for abscissa axis not found."));
							paramMain.modifParamApero().modifUserOrientation().setImage1(tag.getContenu());

							tag = orientation->getEnfant("Point1",&ok);
							if (!ok) return sbase+conv(QObject::tr("First point of abscissa axis not found."));
							text = tag.getContenu();
							bool okx, oky;	
							int x =	text.section(" ",0,0).toInt(&okx);
							int y =	text.section(" ",1,1).toInt(&oky);
							if (!okx || !oky)
								return sbase+conv(QObject::tr("Uncorrect first point of abscissa axis."));						
							paramMain.modifParamApero().modifUserOrientation().setPoint1(QPoint(x,y));

							tag = orientation->getEnfant("Image2",&ok);
							if (!ok) return sbase+conv(QObject::tr("Second image for abscissa axis not found."));
							paramMain.modifParamApero().modifUserOrientation().setImage2(tag.getContenu());

							tag = orientation->getEnfant("Point2",&ok);
							if (!ok) return sbase+conv(QObject::tr("Second point of abscissa axis not found."));
							text = tag.getContenu();
							x = text.section(" ",0,0).toInt(&okx);
							y = text.section(" ",1,1).toInt(&oky);
							if (!okx || !oky)
								return sbase+conv(QObject::tr("Uncorrect second point of abscissa axis."));						
							paramMain.modifParamApero().modifUserOrientation().setPoint2(QPoint(x,y));
						} else
							paramMain.modifParamApero().modifUserOrientation().setBascOnPlan(false);
					}
					//orientation absolue d'une image
					else if (orientMethode==QString("ImageGeoref")) {
						paramMain.modifParamApero().modifUserOrientation() .setOrientMethode(2);

						//Image
						tag = orientation->getEnfant("Image",&ok);
						if (!ok) return sbase+conv(QObject::tr("Fail to find the image of known georeferencing."));
						paramMain.modifParamApero().modifUserOrientation().setImageGeoref(tag.getContenu());

						tag = orientation->getEnfant("ByFile",&ok);
						bool ok2;
						orientation->getEnfant("Centre",&ok2).getContenu();
						if (!ok && !ok2) return sbase+conv(QObject::tr("Fail to find georeferencing of image of %1.")).arg(paramMain.getParamApero().getUserOrientation().getImageGeoref());

						if (ok) {
							text = tag.getContenu();
							paramMain.modifParamApero().modifUserOrientation().setGeorefFile(text);
						} else {
							//centre
							QString centre = orientation->getEnfant("Centre").getContenu();
							for (int i=0; i<3; i++) {
								paramMain.modifParamApero().modifUserOrientation().modifCentreAbs()[i] = centre.section(" ",i,i).toDouble(&ok);
								if (!ok) return sbase+conv(QObject::tr("A coordinate of georeferenced image summit is unvalid."));
							}

							//rotation
							const XmlTag* rotation = &(orientation->getEnfant("Rotation",&ok));
							if (!ok) return sbase+conv(QObject::tr("Fail to find rotation of image %1 georeferencing.")).arg(paramMain.getParamApero().getUserOrientation().getImageGeoref());
							for (int i=0; i<3; i++) {
								tag = rotation->getEnfant(QString("L%1").arg(i+1),&ok);
								if (!ok) return sbase+conv(QObject::tr("Fail to find line L%1 of image %2 georeferencing rotation.")).arg(i+1).arg(paramMain.getParamApero().getUserOrientation().getImageGeoref());
								text = tag.getContenu();
								for (int k=0; k<3; k++) {
									bool ok = false;
									paramMain.modifParamApero().modifUserOrientation().modifRotationAbs()[3*i+k] = text.section(" ",k,k).toDouble(&ok);
									if (!ok) return sbase+conv(QObject::tr("A coordinate of image %1 georeferencing rotation is unvalid.")).arg(paramMain.getParamApero().getUserOrientation().getImageGeoref());
								}
							}
						}
					}
					//échelle
					int om = paramMain.getParamApero().getUserOrientation().getOrientMethode();
					if (om==1 || om==2) {
						tag = orientation->getEnfant("ImagesEchelle",&ok);
						if (!ok && (om!=1 || !paramMain.modifParamApero().modifUserOrientation().getBascOnPlan()))
							return sbase+conv(QObject::tr("Images for rescaling (absolute orientation) not found."));	//échelle obligatoire si orient par img ou si orient manu sans plan+dir
						if (ok) {	//partie pas obligatoire si orient manuelle avec déjà plan+dir
							text = tag.getContenu();
							paramMain.modifParamApero().modifUserOrientation().setFixEchelle(true);
							if (!text.isEmpty()) {
								for (int i=0; i<4; i++)
									paramMain.modifParamApero().modifUserOrientation().modifImages()[i] = text.section(" ",i,i);
							}

							tag = orientation->getEnfant("PointsEchelle",&ok);
							if (!ok) return sbase+conv(QObject::tr("Selected points for rescaling (absolute orientation) not found."));
							text = tag.getContenu();
							for (int i=0; i<4; i++) {
								QVector<int> v(2);
								for (int j=0; j<2; j++) {
									QString s2 = text.section(" ",2*i+j,2*i+j);
									bool b = false;
									v[j] = s2.toInt(&b);
									if (!b) return sbase+conv(QObject::tr("Uncorrect selected point for rescaling (absolute orientation).\n(coordinate %1)")).arg(2*i+j+1);
								}
								paramMain.modifParamApero().modifUserOrientation().modifPoints()[i] = QPoint(v.at(0),v.at(1));
							}

							tag = orientation->getEnfant("Distance",&ok);
							if (!ok) return sbase+conv(QObject::tr("Selected distance for rescaling (absolute orientation) not found."));
							text = tag.getContenu();
							paramMain.modifParamApero().modifUserOrientation().setDistance(text.toInt(&ok));
							if (!ok) return sbase+conv(QObject::tr("Uncorrect distance for rescaling (absolute orientation)."));
						} else
							paramMain.modifParamApero().modifUserOrientation().setFixEchelle(false);
					}
					//orientation par points d'appui
					if (orientMethode==QString("Appuis")) {
						paramMain.modifParamApero().modifUserOrientation() .setOrientMethode(3);

						//points GPS
						tag = orientation->getEnfant("PointsGPS",&ok);
						if (!ok) return sbase+conv(QObject::tr("GCP file for georeferencing not found."));
						paramMain.modifParamApero().modifUserOrientation().setPointsGPS(tag.getContenu());

						//points images
						tag = orientation->getEnfant("AppuisImage",&ok);
						if (!ok) return sbase+conv(QObject::tr("Image measure file of GCP for georeferencing not found."));
						paramMain.modifParamApero().modifUserOrientation().setAppuisImg(tag.getContenu());

					//orientation par sommets GPS
					} else if (orientMethode==QString("Sommets")) {
						paramMain.modifParamApero().modifUserOrientation() .setOrientMethode(4);

						//sommets GPS
						tag = orientation->getEnfant("SommetsGPS",&ok);
						if (!ok) return sbase+conv(QObject::tr("Camera GPS coordinate file not found."));
						paramMain.modifParamApero().modifUserOrientation().setPointsGPS(tag.getContenu());
					}
				}

				if (paramMain.getCurrentMode()==ParamMain::PoseEnCours) {   //PoseEnCours
					tag = xmlTree.getMainTag().getEnfant("Avancement",&ok);
					if (!ok) return sbase+conv(QObject::tr("Computation step not found."));	
					text = tag.getContenu();
					int avancement = text.toInt(&ok);
					if (!ok)
						return sbase+conv(QObject::tr("Uncorrect computation step ."));
					paramMain.setAvancement(avancement);
					if (avancement>=AperoThread::AutoCalibration)
						paramMain.modifParamApero().setFiltrage(false);
				}
				else if (paramMain.getCurrentMode()==ParamMain::PoseMode)   //PoseMode
					paramMain.modifParamApero().setFiltrage(false);

				else {   //CarteEnCours, EndMode
					paramMain.modifParamApero().setFiltrage(false);
					//cartes de profondeur
					const XmlTag* carteTag  = &(xmlTree.getMainTag().getEnfant("Cartes_Profondeur",&ok));
					if (!ok) return sbase+conv(QObject::tr("Depth map parameters not found."));

					//liste des cartes
					QVector<CarteDeProfondeur> listMasques;
					QList<const XmlTag*> listeCartes = carteTag->getEnfants("Carte",&ok);
					if (!ok) return sbase+conv(QObject::tr("No depth maps found."));
					
					for (QList<const XmlTag*>::const_iterator it=listeCartes.begin(); it!=listeCartes.end(); it++) {
						CarteDeProfondeur carte;
						if (paramMain.getCurrentMode()==ParamMain::CarteEnCours)
							carte.setACalculer(true);

						tag = (*it)->getEnfant("Recalculer",&ok);
						if (ok && paramMain.getCurrentMode()==ParamMain::CarteEnCours && tag.getContenu()==QString("non"))
							carte.setACalculer(false);

						tag = (*it)->getEnfant("RefImg",&ok);
						if (!ok) return sbase+conv(QObject::tr("Mask reference image not found (depth map %1)")).arg(it-listeCartes.begin());
						carte.setImageDeReference(tag.getContenu());

						tag = (*it)->getEnfant("ImgsCorrel",&ok);
						if (!ok) return sbase+conv(QObject::tr("Images for correlation not found. (depth map %1)")).arg(it-listeCartes.begin());						
						text = tag.getContenu();
						QString s = text.section(" ",0,0);
						int i = 0;
						while (!s.isEmpty()) {
							carte.modifImagesCorrel().push_back(s);
							i++;
							s = text.section(" ",i,i);
						}

						tag = (*it)->getEnfant("Repere",&ok);	
						if (!ok) return sbase+conv(QObject::tr("Depth map frame not found. (depth map %1)")).arg(it-listeCartes.begin());		
						if (tag.getContenu()!=QString("Image") && tag.getContenu()!=QString("Special") && tag.getContenu()!=QString("Terrain")) return sbase+conv(QObject::tr("First depth map parameters not found (depth map %1)")).arg(it-listeCartes.begin());		
						carte.setRepere(tag.getContenu()==QString("Image"));
						carte.setAutreRepere(tag.getContenu()==QString("Special"));	

						tag = (*it)->getEnfant("DoOrtho",&ok);
						if (ok) carte.setDoOrtho(tag.getContenu()==QString("Oui"));	

						if (carte.getDoOrtho()) {
							tag = (*it)->getEnfant("ImgsOrtho",&ok);
							if (!ok) return sbase+conv(QObject::tr("Images to be orthorectified not found. (depth map %1)")).arg(it-listeCartes.begin());		
							text = tag.getContenu();
							QString s = text.section(" ",0,0);
							int i = 0;
							while (!s.isEmpty()) {
								carte.modifImgsOrtho().push_back(s);
								i++;
								s = text.section(" ",i,i);
							}

							tag = (*it)->getEnfant("EchelleOrtho",&ok);
							if (!ok) return sbase+conv(QObject::tr("Scale of images to be orthorectified not found. (depth map %1)")).arg(it-listeCartes.begin());		
							double ech = tag.getContenu().toDouble(&ok);
							if (!ok || ech==0) return sbase+conv(QObject::tr("Scale of images to be orthorectified is unvalid. (depth map %1)")).arg(it-listeCartes.begin());		
							carte.setEchelleOrtho(ech);
						}

						//fichier de référencement du masque
						/*QString masqImg;
						QString err = FichierMasque::lire(paramMain.getDossier()+carte.getReferencementMasque(),masqImg);
						if (!err.isEmpty())
							return err;
						if (masqImg!=carte.getMasque())
							return sbase+conv(QObject::tr("Le masque %1 ne correspond pas à celui indiqué dans son fichier de référencement.")).arg(carte.getMasque());
						xmlReader.readNext();*/

						tag = (*it)->getEnfant("Intervalle",&ok);
						if (!ok) return sbase+conv(QObject::tr("Correlation search interval not found."));
						text = tag.getContenu();
						float pmin = text.section(" ",0,0).trimmed().simplified().toDouble(&ok);
						if (!ok) return sbase+conv(QObject::tr("Correlation search interval is unvalid (minimum value)."));
						float pmax = text.section(" ",1,1).toDouble(&ok);
						if (!ok) return sbase+conv(QObject::tr("Correlation search interval is unvalid (maximum value)."));
						carte.setInterv( pair<float,float>(pmin,pmax) );

						tag = (*it)->getEnfant("Regul",&ok);
						if (!ok) return sbase+conv(QObject::tr("High slope control parameters not found."));
						text = tag.getContenu();
						if (text.isEmpty())
							carte.setDiscontinuites(false);
						else {
							carte.setDiscontinuites(true);
							bool ok;
							float seuil = text.section(" ",0,0).toDouble(&ok);
							if (!ok) return sbase+conv(QObject::tr("A parameter to manage high slopes is uncorrect (threshold)."));
							float coeff = text.section(" ",1,1).toDouble(&ok);
							if (!ok) return sbase+conv(QObject::tr("A parameter of high slope control is unvalid (weight coefficient)."));
							carte.setSeuilZ( seuil );
							carte.setSeuilZRelatif( coeff );
						}

						listMasques.push_back(carte);
					}

					paramMain.setParamMicmac(listMasques);

					if (paramMain.getCurrentMode()==ParamMain::CarteEnCours) {   //CarteEnCours
						tag = xmlTree.getMainTag().getEnfant("Avancement",&ok);
						if (!ok) return sbase+conv(QObject::tr("Computation step not found."));		
						int avancement = tag.getContenu().toInt(&ok);
						if (!ok) return sbase+conv(QObject::tr("Uncorrect computation step ."));
						paramMain.setAvancement(avancement);
					}
				}
			}
		}
	
		//images		   //ImageMode, PointsEnCours, PointsMode, PoseEnCours, PoseMode, CarteEnCours, EndMode
		const XmlTag* imagesTag = &(xmlTree.getMainTag().getEnfant("Images",&ok));
		if (!ok) return sbase+conv(QObject::tr("Images not found."));	

		paramMain.modifParamApero().modifImgToOri().clear();
		QList<const XmlTag*> listeImgs = imagesTag->getEnfants("Image",&ok);
		if (!ok) return sbase+conv(QObject::tr("No images found.\n"));

		for (QList<const XmlTag*>::const_iterator it=listeImgs.begin(); it!=listeImgs.end(); it++) {
			ParamImage array;
			tag = (*it)->getEnfant("ImageRAW",&ok);
			if (!ok) return sbase+conv(QObject::tr("A raw image is unvalid."));
			array.setImageRAW(tag.getContenu());	

			if (paramMain.getCurrentMode()!=ParamMain::ImageMode) {
				if (paramMain.getCurrentMode()!=ParamMain::PointsEnCours || paramMain.getAvancement()>=PastisThread::Conversion) {
					tag = (*it)->getEnfant("ImageTif",&ok);
					if (!ok) return sbase+conv(QObject::tr("Uncorrect tif image."));
					array.setImageTif(tag.getContenu());
				}
				if (paramMain.getCurrentMode()!=ParamMain::PointsEnCours) {
					paramMain.modifParamApero().modifImgToOri().push_back(array.getImageTif());
					tag = (*it)->getEnfant("Orientable",&ok);
					if (ok && tag.getContenu()==QString("non")) paramMain.modifParamApero().modifImgToOri().pop_back();
				}
			}
			paramMain.modifCorrespImgCalib().push_back(array);
cout << paramMain.getCorrespImgCalib().last().getImageTif().toStdString() << endl;
		}

		//calibrations
		if (paramMain.getCurrentMode()!=ParamMain::ImageMode) {
			const XmlTag* calibTag = &(xmlTree.getMainTag().getEnfant("Calibrations",&ok));
			if (!ok) return sbase+conv(QObject::tr("Calibrations not found."));
			QList<const XmlTag*> listeCalibs = calibTag->getEnfants("Calibration",&ok);
			if (!ok) return sbase+conv(QObject::tr("Calibrations not found."));
			paramMain.modifParamPastis().modifCalibFiles().clear();
			paramMain.modifParamPastis().modifCalibs().clear();

			for (QList<const XmlTag*>::const_iterator it=listeCalibs.begin(); it!=listeCalibs.end(); it++) {
				pair<QString, int> p;
				tag = (*it)->getEnfant("Fichier",&ok);
				if (!ok) return sbase+conv(QObject::tr("Uncorrect calibration file name."));	
				p.first = tag.getContenu();
				tag = (*it)->getEnfant("Focale",&ok);
				if (!ok) return sbase+conv(QObject::tr("Uncorrect calibration focal length."));	
				p.second = tag.getContenu().toInt(&ok);
				if (!ok) return sbase+conv(QObject::tr("Uncorrect number for a calibration focal length."));	
				paramMain.modifParamPastis().modifCalibFiles().push_back(p);
				
				if (paramMain.getCurrentMode()!=ParamMain::PointsEnCours || paramMain.getAvancement()>=PastisThread::PtsInteret) {
				        CalibCam calib;
					QString error(FichierCalibCam::lire(paramMain.getDossier(), p.first, calib));
					if (!error.isEmpty()) return sbase+p.first + QString(" : ") + error;
					calib.setFocale(p.second);
					paramMain.modifParamPastis().modifCalibs().push_back(calib);
				}
			}
		}
	}
	return QString();
}

///////////////////////////////////////////////////////////////////////////////////////////


//liste des caméras utilisées
BDCamera::BDCamera() { BDfile = applicationPath() + QString("/../interface/xml/BDCamera.xml"); }

QString BDCamera::lire (QList<pair<QString,double> >& imgNames) {
	QString sbase = conv(QObject::tr("Camera data base %1 reading :\n")).arg(BDfile);
	XmlTree xmlTree(BDfile);
	QString err = xmlTree.lire();
	if (!err.isEmpty()) return err;

	//étape
	bool ok;
	if (xmlTree.getMainTag().getNom()!=QString("BDCamera")) return sbase+conv(QObject::tr("This file is not a camera data base.")); 

	QList<const XmlTag*> listeCam = xmlTree.getMainTag().getEnfants("Camera",&ok);
	if (!ok) return sbase+conv(QObject::tr("No cameras found.")); 
	
	for (QList<const XmlTag*>::const_iterator it=listeCam.begin(); it!=listeCam.end(); it++) {
		QString nom = (*it)->getAttribut("Nom",&ok);
		if (!ok) return sbase+conv(QObject::tr("A camera has no name.")); 
		QString taillePx = (*it)->getAttribut("Pixel",&ok);
		if (!ok) return sbase+conv(QObject::tr("A camera has no pixel size."));
		double px = 0;
		if (!taillePx.isEmpty()) px = taillePx.toDouble(&ok);
		if (!ok) return conv(QObject::tr("%1 pixel size is not a correct number.")).arg(nom);
		imgNames.push_back(pair<QString,double>(nom,px));
	}

	return QString();
}

bool BDCamera::ecrire (const QList<pair<QString,double> >& imgNames) {
	QFile oldFile(BDfile);
	if (oldFile.exists()) {
		oldFile.remove();
	}

	QFile file(BDfile);
	if (!file.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) {
			return false;
	}
	QXmlStreamWriter xmlWriter(&file);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	xmlWriter.writeStartElement(QString("BDCamera"));
	if (imgNames.count()>0) {
		for (int i=0; i<imgNames.count(); i++) {
			xmlWriter.writeStartElement(QString("Camera"));
			xmlWriter.writeAttribute(QString("Nom"),imgNames.at(i).first);
			xmlWriter.writeAttribute(QString("Pixel"),QVariant(imgNames.at(i).second).toString());
			xmlWriter.writeEndElement();
		}
	}
	xmlWriter.writeEndElement();

	xmlWriter.writeEndDocument();
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////


//liste des caméras utilisées par Micmac
DicoCameraMM::DicoCameraMM(const ParamMain* pMain) : paramMain(pMain) { dicoFile = paramMain->getMicmacDir() + QString("/include/XML_MicMac/DicoCamera.xml"); }

QString DicoCameraMM::ecrire (const CalibCam& calibCam, const QString& img) {
	//lecture de la BD
	QString sbase = conv(QObject::tr("Micmac camera data base %1 reading :\n")).arg(dicoFile);
	XmlTree xmlTree(dicoFile);
	QString err = xmlTree.lire();
	if (!err.isEmpty()) return err;

	//nom de la caméra
	MetaDataRaw metaDataRaw;
	err = focaleRaw(img, *paramMain, metaDataRaw);
	if (!err.isEmpty()) return err;
	if (metaDataRaw.focale==0 || metaDataRaw.imgSize.width()==0 || metaDataRaw.imgSize.height()==0) return sbase+conv(QObject::tr("The metadata extracted from image %1 are unvalid")).arg(img);
	QPointF taille(calibCam.getTaillePx()*metaDataRaw.imgSize.width()/1000.0 , calibCam.getTaillePx()*metaDataRaw.imgSize.height()/1000.0);	//on suppose que le pixel est carré

	//si ce nom existe déjà, rien à faire
	const QList<const XmlTag*> l = xmlTree.getMainTag().getEnfants("CameraEntry");
	for (QList<const XmlTag*>::const_iterator it=l.begin(); it!=l.end(); it++) {
		if ((*it)->getEnfant("Name").getContenu()==metaDataRaw.camera ||(*it)->getEnfant("ShortName").getContenu()==metaDataRaw.camera)
			return QString();		
	}

	//ajout
	XmlTag cameraEntry(QString("CameraEntry"),&xmlTree.modifMainTag());
	cameraEntry.addEnfant(XmlTag(QString("Name"),metaDataRaw.camera,&cameraEntry));
	cameraEntry.addEnfant(XmlTag(QString("SzCaptMm"),QString("%1 %2").arg(taille.x()).arg(taille.y()),&cameraEntry));
	cameraEntry.addEnfant(XmlTag(QString("ShortName"),metaDataRaw.camera,&cameraEntry));
	xmlTree.modifMainTag().addEnfant(cameraEntry);
	if (!xmlTree.ecrire()) return sbase+ conv(QObject::tr("Fail to modify Micmac camera data base"));	//à progr

	return QString();
}

///////////////////////////////////////////////////////////////////////////////////////////


//écriture des fichiers de calibration interne des caméras utilisées
FichierCalibCam::FichierCalibCam() {}

bool FichierCalibCam::ecrire (const QString& dossier, const CalibCam& calibCam) {
	QFile oldFile(dossier + QString("/") + calibCam.getFile());
	if (oldFile.exists()) {
		oldFile.remove();
	}

	QFile newFile(dossier + QString("/") + calibCam.getFile());
	if (!newFile.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) {
			return false;
	}
	QXmlStreamWriter xmlWriter(&newFile);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	xmlWriter.writeStartElement(QString("ExportAPERO"));
	xmlWriter.writeStartElement(QString("CalibrationInternConique"));

		xmlWriter.writeTextElement(QString("KnownConv"),QString("eConvApero_DistM2C"));
		xmlWriter.writeTextElement(QString("PP"),QVariant(calibCam.getPPA().x()).toString()+QString(" ")+QVariant(calibCam.getPPA().y()).toString());
		xmlWriter.writeTextElement(QString("F"),QVariant(calibCam.getFocalePx()).toString());
		xmlWriter.writeTextElement(QString("SzIm"),QVariant(calibCam.getSizeImg().width()).toString()+QString(" ")+QVariant(calibCam.getSizeImg().height()).toString());

		if (calibCam.getType()==0) {
			xmlWriter.writeStartElement(QString("CalibDistortion"));
			xmlWriter.writeStartElement(QString("ModRad"));
				xmlWriter.writeTextElement(QString("CDist"),QVariant(calibCam.getPPS().x()).toString()+QString(" ")+QVariant(calibCam.getPPS().y()).toString());
				for (int i=0; i<4; i++) {
					xmlWriter.writeTextElement(QString("CoeffDist"),QVariant(calibCam.getDistorsion()[i]).toString());
				}
			xmlWriter.writeEndElement();	//ModRad
		} else {
			xmlWriter.writeTextElement(QString("RayonUtile"),QVariant(calibCam.getRayonUtile()).toString());
			xmlWriter.writeStartElement(QString("CalibDistortion"));
			xmlWriter.writeStartElement(QString("ModUnif"));
			xmlWriter.writeTextElement(QString("TypeModele"),QString("eModele_FishEye_10_5_5"));
			for (int i=0; i<calibCam.getParamRadial().count(); i++)
				xmlWriter.writeTextElement(QString("Params"),QVariant(calibCam.getParamRadial().at(i)).toString());
			for (int i=calibCam.getParamRadial().count(); i<40; i++)
				xmlWriter.writeTextElement(QString("Params"),QString("0"));			
			xmlWriter.writeTextElement(QString("Etats"),QVariant(2409).toString());	// ???????????????????????????????????????????????
			xmlWriter.writeEndElement();	//ModUnif
		}
		xmlWriter.writeEndElement();	//CalibDistortion

	xmlWriter.writeEndElement();	//CalibrationInternConique
	xmlWriter.writeEndElement();	//ExportAPERO

	xmlWriter.writeEndDocument();
	return true;
}

QString FichierCalibCam::lire(const QString& dossier, const QString& fichier, CalibCam& calibCam, int focalemm) {
	QString sbase = conv(QObject::tr("Calibration file %1 reading :\n")).arg(dossier+fichier);
	XmlTree xmlTree(dossier+fichier);
	QString err = xmlTree.lire();
	if (!err.isEmpty()) return err;

	bool ok;
	QString text;
	XmlTag tag0;
	XmlTag& tag = *(&tag0);
	const XmlTag* calibTag;
	if (xmlTree.getMainTag().getNom()!=QString("ExportAPERO")) {
		if (xmlTree.getMainTag().getNom()!=QString("CalibrationInternConique")) return sbase+conv(QObject::tr("This file is not a camera calibration file."));
		else calibTag = &(xmlTree.getMainTag());
	} else {
		calibTag = &(xmlTree.getMainTag().getEnfant("CalibrationInternConique",&ok));
		if (!ok) return sbase+conv(QObject::tr("Missing element BDCamera."));
	}

	//point principal
	tag = calibTag->getEnfant("PP",&ok);
	if (!ok) return sbase+conv(QObject::tr("Autocollimation main point not found."));
	text = tag.getContenu().trimmed();
	double PPAx  = text.section(" ",0,0).toDouble(&ok);
	if (!ok) return sbase+conv(QObject::tr("Uncorrect autocollimation main point coordinates."));
	double PPAy  = text.section(" ",1,1).toDouble(&ok);
	if (!ok) return sbase+conv(QObject::tr("Uncorrect autocollimation main point coordinates."));

	//focale
	tag = calibTag->getEnfant("F",&ok);
	if (!ok) return sbase+conv(QObject::tr("Focal length not found."));
	double f = tag.getContenu().trimmed().toDouble(&ok);
	if (!ok) return sbase+conv(QObject::tr("Uncorrect focal length."));

	//taille de l'image
	tag = calibTag->getEnfant("SzIm",&ok);
	if (!ok) return sbase+conv(QObject::tr("Image size not found."));
	text = tag.getContenu().trimmed();
	double w  = text.section(" ",0,0).toDouble(&ok);
	if (!ok) return sbase+conv(QObject::tr("Uncorrect image size."));
	double h  = text.section(" ",1,1).toDouble(&ok);
	if (!ok) return sbase+conv(QObject::tr("Uncorrect image size."));

	//type
	int type;
	tag = calibTag->getEnfant("RayonUtile",&ok);
	if (ok) type = 1;
	else {
		tag = calibTag->getEnfant("CalibDistortion",&ok);
		if (ok) type = 0;
		else return sbase+conv(QObject::tr("Distorsion not found."));
	}

	//rayon utile
	const XmlTag* distTag;
	double rayon = 0;
	if (type==1) {
		rayon  = tag.getContenu().section(" ",0,0).toDouble(&ok);
		if (!ok) return sbase+conv(QObject::tr("Uncorrect efficient radius."));
		tag = calibTag->getEnfant("CalibDistortion",&ok);
		if (!ok) return sbase+conv(QObject::tr("Distorsion not found."));
		distTag = &(tag.getEnfant("ModUnif",&ok));
		if (!ok) return sbase+conv(QObject::tr("Mode not found."));		
	} else {
		distTag = &(tag.getEnfant("ModRad",&ok));
		if (!ok) return sbase+conv(QObject::tr("Mode not found."));
	}

	//distorsion
	double PPSx(0), PPSy(0);
	if (type==0) {
		//centre de distorsion
		tag = distTag->getEnfant("CDist",&ok);
		if (!ok) return sbase+conv(QObject::tr("Distorsion center not found."));
		text = tag.getContenu().trimmed();
		PPSx  = text.section(" ",0,0).toDouble(&ok);
		if (!ok) return sbase+conv(QObject::tr("Uncorrect distorsion center coordinates."));
		PPSy  = text.section(" ",1,1).toDouble(&ok);
		if (!ok) return sbase+conv(QObject::tr("Uncorrect distorsion center coordinates."));
	}
		
	//paramètres de distorsion
	QList<const XmlTag*> distListe;
	QVector<double> paramDist;
	if (type==0) {
		distListe = distTag->getEnfants("CoeffDist",&ok);
		paramDist.fill(0,4);
	} else {
		distListe = distTag->getEnfants("Params",&ok);
		paramDist.fill(0,distListe.count());
	}
	if (!ok) return sbase+conv(QObject::tr("No distorsion coefficients found."));
	for (int i=0; i<min(paramDist.count(),distListe.count()); i++) {
		paramDist[i]  = distListe.at(i)->getContenu().trimmed().toDouble(&ok);
		if (!ok) return sbase+conv(QObject::tr("Uncorrect distorsion coefficient."));
	}

	double taillepx = 1.0;
	if (focalemm!=0) {
		taillepx = double(f*1000) / double(focalemm);
		f = focalemm;
	} else
		f /= 1000.0;
	
	calibCam = CalibCam(type,fichier, f, taillepx, QPointF(PPAx,PPAy), QSize(w,h), QPointF(PPSx,PPSy), paramDist, rayon, paramDist);
	return QString();
}

///////////////////////////////////////////////////////////////////////////////////////////

FichierParamImage::FichierParamImage() {}

bool FichierParamImage::ecrire(const QString& fichier, const QVector<ParamImage>& lstImg) {
//écriture de la liste des images
	QFile oldFile(fichier);
	if (oldFile.exists()) {
		oldFile.remove();
	}
	QFile file(fichier);
	if (!file.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) {
			return false;
	}
	QXmlStreamWriter xmlWriter(&file);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	xmlWriter.writeStartElement(QString("KeyedSetsOfNames"));
	xmlWriter.writeStartElement(QString("Sets"));

	for (QVector<ParamImage>::const_iterator it=lstImg.begin(); it!=lstImg.end(); it++) {
		QString img = QString("^/") + (it->getImageTif());
		xmlWriter.writeTextElement(QString("PatternAccepteur"), img);
	}
	xmlWriter.writeEndElement();	//Sets

	xmlWriter.writeTextElement(QString("Key"), QString("Key-Set-All-Im"));

	xmlWriter.writeEndElement();	//KeyedSetsOfNames
	xmlWriter.writeEndDocument();
	
	return true;
}

QString FichierParamImage::lire(const QString& fichier, QVector<ParamImage>& lstImg) {
    QString sbase = conv(QObject::tr("Image list reading %1 : ")).arg(fichier);

	XmlTree xmlTree(fichier);
	QString err = xmlTree.lire();
	if (!err.isEmpty()) return err;

	QString text;
	bool ok;
	XmlTag tag0; 
	XmlTag& tag = *(&tag0);

	if (xmlTree.getMainTag().getNom()!=QString("KeyedSetsOfNames")) return sbase+conv(QObject::tr("This file is not a list of images."));
	
	tag = xmlTree.getMainTag().getEnfant("Sets",&ok);
	if (!ok) return sbase+conv(QObject::tr("No image sets found."));

	const QList<const XmlTag*> imgListe = tag.getEnfants("PatternAccepteur",&ok);
	if (!ok) return sbase+conv(QObject::tr("No images found."));
	if (imgListe.count()<lstImg.count()) return sbase+conv(QObject::tr("Not enough images."));
	if (imgListe.count()>lstImg.count()) return sbase+conv(QObject::tr("Too many images."));

	for (QList<const XmlTag*>::const_iterator it=imgListe.begin(); it!=imgListe.end(); it++) {
		text = (*it)->getContenu().section("/",-1,-1);
		ok = false;
		for (QVector<ParamImage>::iterator it2=lstImg.begin(); it2!=lstImg.end(); it2++) {
			if (text.contains(it2->getImageRAW().section(".",0,-2))) {
	//			it2->setImageTif(text);	
				ok = true;			
				break;
			}
		}
		if (!ok) return sbase+conv(QObject::tr("Image %1 does not match any raw image.")).arg(text);		
	}
	return QString();
}

///////////////////////////////////////////////////////////////////////////////////////////////////


//écriture des couples d'images
FichierCouples::FichierCouples() {}

bool FichierCouples::ecrire (const QString& fichier, const QList<pair<QString, QString> >& couples, const QVector<ParamImage>& rawToTif) {
//écriture des couples dans fichier ; si rawToTif!=0, convertit les noms en tif
	QFile oldFile(fichier);
	if (oldFile.exists()) {
		oldFile.remove();
	}

	QFile newFile(fichier);
	if (!newFile.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) {
			return false;
	}
	QXmlStreamWriter xmlWriter(&newFile);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	xmlWriter.writeStartElement(QString("KeyedSetsORels"));
	xmlWriter.writeStartElement(QString("Sets"));
	xmlWriter.writeStartElement(QString("RelByGrapheExpl"));

	for (QList<pair<QString, QString> >::const_iterator it=couples.begin(); it!=couples.end(); it++) {
		QString img1 = (rawToTif.count()>0) ? traduire(it->first, rawToTif, true) : it->first;
		QString img2 = (rawToTif.count()>0) ? traduire(it->second, rawToTif, true) : it->second;
		if (img1.isEmpty() || img2.isEmpty())
			return false;
		xmlWriter.writeTextElement(QString("Cples"), img1 +QString(" ") + img2);
		xmlWriter.writeTextElement(QString("Cples"), img2 +QString(" ") + img1);
	}

	xmlWriter.writeEndElement();	//RelByGrapheExpl
	xmlWriter.writeEndElement();	//Sets
	xmlWriter.writeTextElement(QString("Key"), QString("Key-Rel-All-Cple"));
	xmlWriter.writeEndElement();	//KeyedSetsORels

	xmlWriter.writeEndDocument();
	return true;
}

QString FichierCouples::convertir(const QString& fichier, const QVector<ParamImage>& rawToTif) {
//réécrit les couples d'images avec les nouveaux noms
	QList<pair<QString, QString> > couples;
	QString tempoFile("tempofile");
	QString err = lire (fichier, couples, rawToTif, false);	//lecture sans traduction
	if (!err.isEmpty())
		return err;
	bool res = ecrire (tempoFile, couples, rawToTif);	//écriture en tif
	if (!res)
		return conv(QObject::tr("Fail to over write image pairs."));
	QFile(fichier).remove();
	QFile(tempoFile).rename(fichier);
	return QString();
}

QString FichierCouples::traduire(const QString& image, const QVector<ParamImage>& rawToTif, bool toTif) {
//convertit un nom d'image en tif si toTif (et teste pour toutes ses formes), en raw sinon (la forme initiale)
	for (int i=0; i<rawToTif.count(); i++) {
		if (toTif && (image==rawToTif.at(i).getImageRAW() || image==rawToTif.at(i).getImageTif()))
			return rawToTif.at(i).getImageTif();
		else if (!toTif && (image==rawToTif.at(i).getImageRAW() || image==rawToTif.at(i).getImageTif()))
			return rawToTif.at(i).getImageRAW();
	}
	return QString();
}

QString FichierCouples::lire (const QString& fichier, QList<pair<QString, QString> >& couples, const QVector<ParamImage>& tifToRaw, bool raw) {
//lit les couples d'images dans fichier et donne leur forme raw (forme initiale) si raw
	QString sbase = conv(QObject::tr("Image pair file %1 reading : ")).arg(fichier);

	XmlTree xmlTree(fichier);
	QString err = xmlTree.lire();
	if (!err.isEmpty()) return err;

	QString text;
	bool ok;
	XmlTag tag0; 
	XmlTag& tag = *(&tag0);

	if (xmlTree.getMainTag().getNom()!=QString("KeyedSetsORels")) return sbase+conv(QObject::tr("This file is not a list of image pairs."));
	
	tag = xmlTree.getMainTag().getEnfant("Sets",&ok);
	if (!ok) return sbase+conv(QObject::tr("No sets of image pairs found."));
	
	tag = tag.getEnfant("RelByGrapheExpl",&ok);
	if (!ok) return sbase+conv(QObject::tr("No method found."));

	const QList<const XmlTag*> cplsListe = tag.getEnfants("Cples",&ok);
	if (!ok) return sbase+conv(QObject::tr("No image pairs found."));

	for (QList<const XmlTag*>::const_iterator it=cplsListe.begin(); it!=cplsListe.end(); it++) {
		text = (*it)->getContenu();
		QString img1 = text.section(" ",0,0);
		QString img2 = text.section(" ",1,1);
		pair<QString, QString> p;
		if (raw) p = pair<QString, QString>( traduire(img1,tifToRaw,false), traduire(img2,tifToRaw,false) );
		else p = pair<QString,QString>(img1,img2);
		if (!couples.contains(p) && !couples.contains(pair<QString,QString>(p.second,p.first))) {
			if (p.first<p.second) couples.push_back(p);
			else if (p.first>p.second) couples.push_back(pair<QString,QString>(p.second,p.first));
		}	
	}
	return QString();
}

///////////////////////////////////////////////////////////////////////////////////////////


FichierAssocCalib::FichierAssocCalib() {}

bool FichierAssocCalib::ecrire (const QString& fichier, const QVector<ParamImage>& assoc) {
	if (assoc.count()==0)
		return false;

	QFile oldFile(fichier);
	if (oldFile.exists()) {
		oldFile.remove();
	}

	QFile newFile(fichier);
	if (!newFile.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) {
			return false;
	}
	QXmlStreamWriter xmlWriter(&newFile);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	xmlWriter.writeStartElement(QString("KeyedNamesAssociations"));

	for (QVector<ParamImage>::const_iterator it=assoc.begin(); it!=assoc.end(); it++) {
		xmlWriter.writeStartElement(QString("Calcs"));
			xmlWriter.writeTextElement(QString("Arrite"), QString("1 1"));
			xmlWriter.writeStartElement(QString("Direct"));
				xmlWriter.writeTextElement(QString("PatternTransform"), QString("^")+it->getImageTif());
				xmlWriter.writeTextElement(QString("CalcName"), it->getCalibration());
			xmlWriter.writeEndElement();	//Direct
		xmlWriter.writeEndElement();	//Calcs
	}
	xmlWriter.writeTextElement(QString("Key"), QString("Key-Assoc-CalibOfIm"));

	xmlWriter.writeEndElement();	//KeyedNamesAssociations
	xmlWriter.writeEndDocument();
	return true;
}

QString FichierAssocCalib::lire (const QString& fichier, QVector<ParamImage> & assoc) {
	QString sbase = conv(QObject::tr("Image - calibration matches %1 reading : ")).arg(fichier);

	XmlTree xmlTree(fichier);
	QString err = xmlTree.lire();
	if (!err.isEmpty()) return err;

	QString text;
	bool ok;
	XmlTag tag0; 
	XmlTag& tag = *(&tag0);
	const XmlTag* assocTag;

	if (xmlTree.getMainTag().getNom()!=QString("KeyedNamesAssociations")) return sbase+conv(QObject::tr("This file is not an image - calibration matche list."));

	const QList<const XmlTag*> assocListe = xmlTree.getMainTag().getEnfants("Calcs",&ok);
	if (!ok) return sbase+conv(QObject::tr("No matches found."));

	for (QList<const XmlTag*>::const_iterator it=assocListe.begin(); it!=assocListe.end(); it++) {
		assocTag = &((*it)->getEnfant("Direct",&ok));
		if (!ok) return sbase+conv(QObject::tr("A match is unvalid."));

		tag = assocTag->getEnfant("PatternTransform",&ok);
		if (!ok) return sbase+conv(QObject::tr("The image of a match is not found."));
		text = tag.getContenu();
		text = text.right(text.size()-1);
		int idx = -1;
		for (QVector<ParamImage>::iterator it2=assoc.begin(); it2!=assoc.end(); it2++) {
			if (text==it2->getImageTif()) {
				idx = it2-assoc.begin();
				break;
			}
		}
		if (idx==-1) return sbase+conv(QObject::tr("Unknown tif image %1.")).arg(text);
		
		tag = assocTag->getEnfant("CalcName",&ok);
		if (!ok) return sbase+conv(QObject::tr("Fail to find the calibration of a matching (%1).")).arg(text);
		text = tag.getContenu();
		if (text.contains("/")) text = text.section("/",-1,-1);
		assoc[idx].setCalibration(text);
	}

	return QString();
}

///////////////////////////////////////////////////////////////////////////////////////////


QString FichierFiltrage::extensionSortie = QString("filtre");
FichierFiltrage::FichierFiltrage() {}

bool FichierFiltrage::ecrire (QString fichier, QString dossier, QSize imgSize) {
	QFile oldFile(dossier+fichier);
	if (oldFile.exists()) {
		oldFile.remove();
	}

	QFile newFile(dossier+fichier);
	if (!newFile.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) {
			return false;
	}
	QXmlStreamWriter xmlWriter(&newFile);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	xmlWriter.writeStartElement(QString("ParamFusionSift"));
		xmlWriter.writeTextElement(QString("dossierImg"), QString());
		xmlWriter.writeTextElement(QString("dossier"), dossier);
		xmlWriter.writeTextElement(QString("extensionSortie"), extensionSortie);
		xmlWriter.writeTextElement(QString("box"), QString("0 0 ")+QVariant(imgSize.width()).toString()+QString(" ")+QVariant(imgSize.height()).toString());
		xmlWriter.writeTextElement(QString("filtre2"), QString("false"));
	xmlWriter.writeEndElement();	//ParamFusionSift
	xmlWriter.writeEndDocument();
	return true;
}

void FichierFiltrage::renameFiles (QString dossier) {
	QDir dir(dossier+"Homol");
	QStringList lD = dir.entryList(QDir::AllDirs);
	for (int j=0; j<lD.count(); j++) {
		QStringList l = QDir(dossier+"Homol/"+lD.at(j)).entryList(QStringList("*.dat"), QDir::Files);
		for (int i=0; i<l.count(); i++) {
			if (l.at(i).contains(extensionSortie)) continue;
			QFile(dossier+"Homol/"+lD.at(j)+"/"+l.at(i)).rename(dossier+"Homol/"+lD.at(j)+"/"+l.at(i).section(".",0,-2)+"_init.dat");
		}
		for (int i=0; i<l.count(); i++) {
			if (!l.at(i).contains(extensionSortie)) continue;
			QString nom = l.at(i);
			QFile(dossier+"Homol/"+lD.at(j)+"/"+l.at(i)).rename(dossier+"Homol/"+lD.at(j)+"/"+nom.remove(QString("_")+extensionSortie));
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////////////


FichierMaitresse::FichierMaitresse() {}

bool FichierMaitresse::ecrire (QString fichier, QString maitresse, QString calibFile, bool sommetsGPS) {
	QFile oldFile(fichier);
	if (oldFile.exists()) {
		oldFile.remove();
	}

	QFile newFile(fichier);
	if (!newFile.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) {
			return false;
	}
	QXmlStreamWriter xmlWriter(&newFile);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	xmlWriter.writeTextElement(QString("PatternName"), maitresse);
	bool ok = false;
	int focale = calibFile.section(".",0,-2).right(3).toInt(&ok);
	if (!ok) {
		cout << QObject::tr("Fail to extract focal length from master image calibration file name.").toStdString() << endl;
		return false;
	}
	QString f = QVariant(focale).toString();
	while (f.count()<3)
		f = QString("0")+f;
	xmlWriter.writeTextElement(QString("CalcNameCalib"), QString("TheKeyCalib_")+f);
	if (sommetsGPS) xmlWriter.writeTextElement(QString("IdBDCentre"), QString("Id-Centre"));
	xmlWriter.writeEndDocument();
	return true;
}

QString FichierMaitresse::lire (QString fichier, QString & maitresse) {
	QString sbase = conv(QObject::tr("Master image file %1 reading : ")).arg(fichier);

	XmlTree xmlTree(fichier);
	QString err = xmlTree.lire(false);
	if (!err.isEmpty()) return err;

	if (xmlTree.getMainTag().getNom()!=QString("PatternName")) return sbase+conv(QObject::tr("Master image not found."));
	maitresse = xmlTree.getMainTag().getContenu();

	return QString();
}

///////////////////////////////////////////////////////////////////////////////////////////


FichierExportPly::FichierExportPly() {}

bool FichierExportPly::ecrire (QString fichier) {
	QFile oldFile(fichier);
	if (oldFile.exists()) oldFile.remove();

	QFile newFile(fichier);
	if (!newFile.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) {
			return false;
	}
	QXmlStreamWriter xmlWriter(&newFile);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	xmlWriter.writeStartElement(QString("ExportNuage"));
		xmlWriter.writeTextElement(QString("NameOut"), QString("Cameras.ply"));
		xmlWriter.writeTextElement(QString("PlyModeBin"), QString("true"));
		xmlWriter.writeTextElement(QString("NameRefLiaison"), QString("Id_Pastis_Hom"));
		xmlWriter.writeStartElement(QString("Pond"));
			xmlWriter.writeTextElement(QString("EcartMesureIndiv"), QString("1.0"));
			xmlWriter.writeTextElement(QString("EcartMax"), QString("1000.4"));
		xmlWriter.writeEndElement();	//Pond
		xmlWriter.writeTextElement(QString("KeyFileColImage"), QString("NKS-Assoc-Id"));	//Key-Assoc-Im2OrFinale, Key-Assoc-Im2OrFinale-3, Key-Assoc-Im2OrFinale
		xmlWriter.writeStartElement(QString("NuagePutCam"));
			xmlWriter.writeTextElement(QString("ColCadre"), QString("255 0 0"));
			xmlWriter.writeTextElement(QString("ColRay"), QString("0 255 0"));
			xmlWriter.writeTextElement(QString("Long"), QString("0.3"));
			xmlWriter.writeTextElement(QString("StepSeg"), QString("0.01"));
		xmlWriter.writeEndElement();	//NuagePutCam
	xmlWriter.writeEndElement();	//ExportNuage

	xmlWriter.writeEndDocument();
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////


FichierImgToOri::FichierImgToOri() {}

bool FichierImgToOri::ecrire (const QString& fichier, const QStringList& imgOri, const QVector<ParamImage>& assoc, const QString& maitresse, const QList<std::pair<QString, int> >& calibFiles, bool withGPSSummit, bool monoechelle, int etape1, const QList<int>& posesFigees) {
	if (!monoechelle && etape1!=2 && (posesFigees.count()==0))
		return false;
	QFile oldFile(fichier);
	if (oldFile.exists()) {
		oldFile.remove();
	}

	QFile newFile(fichier);
	if (!newFile.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) {
		return false;
	}
	QXmlStreamWriter xmlWriter(&newFile);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	for (int k=0; k<calibFiles.count(); k++) {
		//vérification de la présence d'images à orienter correspondant à cette calibration
		if (calibFiles.count()>1) {
			bool b = false;
			for (int i=0; i<assoc.size(); i++) {
				if (assoc.at(i).getCalibration()==calibFiles.at(k).first && imgOri.indexOf(assoc.at(i).getImageTif())!=-1) {
					b = true;
					break;
				}
			}
			if (!b) continue;
			if (!monoechelle && etape1==0 && !posesFigees.contains(calibFiles.at(k).second))
				continue;	//pas de focale longue à l'étape 1 du multi-échelle
		}

		//images à orienter de cette focale
		QStringList l;
		if (calibFiles.count()>1) {
			for (int i=0; i<assoc.size(); i++) {
				if (imgOri.indexOf(assoc.at(i).getImageTif())==-1) continue;	//img non orientable (pas de points de liaison)
				if (assoc.at(i).getImageTif()==maitresse && (monoechelle || etape1==0)) continue;	//l'image maîtresse n'est pas orientée, sauf dans l'étape2 multi-f (l'image maîtresse a une focale courte) et dans l'étape 3
				if (assoc.at(i).getCalibration()!=calibFiles.at(k).first) continue;	//ordonnées par calibration
				l.push_back(assoc.at(i).getImageTif());
			}
			if (l.count()==0) continue;
		} else
			l = imgOri;

		xmlWriter.writeStartElement(QString("PoseCameraInc"));
			for (int i=0; i<l.count(); i++)
				xmlWriter.writeTextElement(QString("PatternName"), l.at(i));
			xmlWriter.writeTextElement(QString("AutoRefutDupl"), QVariant(true).toString());	//supprime les duplicatats

			if (etape1!=2 && (monoechelle || etape1==0 || !posesFigees.contains(calibFiles.at(k).second)))
				xmlWriter.writeTextElement(QString("InitNow"), QString("false"));	//les focales courtes ont une valeur initiale à l'étape 2 du multi-échelle

			QString f = QVariant(calibFiles.at(k).second).toString();
			while (f.count()<3)
				f = QString("0")+f;
			xmlWriter.writeTextElement(QString("CalcNameCalib"), QString("TheKeyCalib_")+f);

			if (withGPSSummit)
				xmlWriter.writeTextElement(QString("IdBDCentre"), QString("Id-Centre"));

			if (etape1!=2 && (monoechelle || etape1==0 || posesFigees.contains(calibFiles.at(k).second)))	//pour les focales courtes
				xmlWriter.writeTextElement(QString("PosesDeRattachement"), QString("0"));

			if (etape1!=2 && (monoechelle || etape1==0 || !posesFigees.contains(calibFiles.at(k).second))) {
				xmlWriter.writeStartElement(QString("MEP_SPEC_MST"));
					xmlWriter.writeTextElement(QString("Show"), QString("true"));
					if (!monoechelle && etape1==1)
						xmlWriter.writeTextElement(QString("MontageOnInit"), QString("false"));//f longues
				xmlWriter.writeEndElement();	//MEP_SPEC_MST

				xmlWriter.writeStartElement(QString("PosValueInit"));
					xmlWriter.writeStartElement(QString("PoseFromLiaisons"));
						xmlWriter.writeStartElement(QString("LiaisonsInit"));
							xmlWriter.writeTextElement(QString("NameCam"), QString("###"));
							xmlWriter.writeTextElement(QString("IdBD"), QString("Id_Pastis_Hom"));
						xmlWriter.writeEndElement();	//LiaisonsInit
					xmlWriter.writeEndElement();	//PoseFromLiaisons
				xmlWriter.writeEndElement();	//PosValueInit
			} else {
				xmlWriter.writeStartElement(QString("PosValueInit"));	//init des foc courtes par les résult de l'étape 1 (multi-éch), et de toutes les imgs pour l'étape 3 par les résult de l'étape 2
					xmlWriter.writeTextElement(QString("PosFromBDOrient"), QString("Key-Ori-Init"));
				xmlWriter.writeEndElement();	//PosValueInit
			}
		xmlWriter.writeEndElement();	//PoseCameraInc
	}
	xmlWriter.writeEndDocument();
	return true;
}

bool FichierImgToOri::ecrire (const QString& fichier, const QStringList& imgOri, const QString& maitresse, const QString& calibFile, int focale) {
	QList<pair<QString, int> > l;
	l.append(pair<QString, int>(calibFile,focale));
	return ecrire(fichier, imgOri, QVector<ParamImage>(0), maitresse, l, false, true, true, QList<int>());
}

QString FichierImgToOri::lire(const QString& fichier, QStringList& imgOri, const QVector<ParamImage>& assoc) {
	QFile file(fichier);
	if (!file.open(QFile::ReadOnly | QFile::Text)) {
			return conv(QObject::tr("Fail to open file of images to be oriented."));
	}
	QXmlStreamReader xmlReader(&file);

	while (!xmlReader.atEnd()) {
		xmlReader.readNext();
		while (!xmlReader.isStartElement()) {
			xmlReader.readNext();
			if (xmlReader.atEnd())
				break;
		}
		
		if (xmlReader.atEnd() || xmlReader.name().toString() != QString("PatternName"))
			continue;

		QString text = xmlReader.readElementText().trimmed();
		//on vérifie que c'est bien une image
		bool isImg = false;
		for (int i=0; i<assoc.count(); i++) {
			if (assoc.at(i).getImageTif()==text) {
				isImg = true;
				break;
			}
		}
		if (!isImg)
			return conv(QObject::tr("Image to be oriented %1 does not belong to loaded image list.")).arg(text);
		imgOri.push_back(text);
	}

	
	return qtError(xmlReader, conv(QObject::tr("Mask referencing file reading : ")));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


FichierDefCalibration::FichierDefCalibration() {}

bool FichierDefCalibration::ecrire (const QString& fichier, const QList<std::pair<QString, int> >& calibFiles, bool monoechelle, int etape, const QList<int>& calibFigees) {
	if (!monoechelle && calibFiles.count()==0)
		return false;
	QFile oldFile(fichier);
	if (oldFile.exists()) {
		oldFile.remove();
	}
	QFile newFile(fichier);
	if (!newFile.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) {
		return false;
	}
	QXmlStreamWriter xmlWriter(&newFile);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	for (int i=0; i<calibFiles.count(); i++) {
		if (!monoechelle && etape==0 && !calibFigees.contains(calibFiles.at(i).second)) continue;	//pas de f lges à l'étape1 (multi-foc)
		xmlWriter.writeStartElement(QString("CalibrationCameraInc"));
			QString f = QVariant(calibFiles.at(i).second).toString();
			while (f.count()<3)
				f = QString("0")+f;
			xmlWriter.writeTextElement(QString("Name"), QString("TheKeyCalib_")+f);
			xmlWriter.writeStartElement(QString("CalValueInit"));
				xmlWriter.writeStartElement(QString("CalFromFileExtern"));
					if (!monoechelle && etape!=0 && (calibFigees.contains(calibFiles.at(i).second) || etape==2)) {
						QString f = QVariant(calibFiles.at(i).second).toString();
						while (f.count()<3)
							f = QVariant(0).toString() + f;
						if (etape==2)
							xmlWriter.writeTextElement(QString("NameFile"), QString("Ori-F/F")+f+QString("_AutoCalFinale.xml"));	//fichier résultat de l'étape 2 pr toutes les imgs
						else
							xmlWriter.writeTextElement(QString("NameFile"), QString("Orient/F")+f+QString("_AutoCalInit.xml"));	//fichier résultat de l'étape 1 pour les foc courtes
					} else
						xmlWriter.writeTextElement(QString("NameFile"), calibFiles.at(i).first);	//fichier utilisateur
					xmlWriter.writeTextElement(QString("NameTag"), QString("CalibrationInternConique"));
				xmlWriter.writeEndElement();	//CalFromFileExtern
			xmlWriter.writeEndElement();	//CalValueInit
			if (!monoechelle && etape==2 && !(calibFigees.contains(calibFiles.at(i).second))) {	//dissociation des calib pour l'étape 3
				xmlWriter.writeStartElement(QString("CalibPerPose"));
					xmlWriter.writeTextElement(QString("KeyPose2Cal"), QString("Key-Assoc-Cal-Var"));
				xmlWriter.writeEndElement();	//CalibPerPose
			}
		xmlWriter.writeEndElement();	//CalibrationCameraInc
	}

	xmlWriter.writeEndDocument();
	return true;
}

bool FichierDefCalibration::ecrire (const QString& fichier, const QList<std::pair<QString, int> >& calibFiles, const QVector<bool>& calibDissoc) {
	QList<int> calibFigees;
	for (int i=0; i<calibFiles.count(); i++) {
		if (!calibDissoc.at(i))
			calibFigees.push_back(calibFiles.at(i).second);
	}
	bool b = ecrire (fichier, calibFiles, false, 2, calibFigees);
	return b;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//orientation suivant un plan et une axe saisis par l'utilisateur
FichierBasculOri::FichierBasculOri() {}

bool FichierBasculOri::ecrire (const QString& fichier, const UserOrientation& param, const QStringList& images) {
	QFile oldFile(fichier);
	if (oldFile.exists()) {
		oldFile.remove();
	}
	QFile newFile(fichier);
	if (!newFile.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) {
		return false;
	}

	if (param.getOrientMethode()==0) {
		newFile.close();
		return true;
	}

	QXmlStreamWriter xmlWriter(&newFile);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	//basculement (plan)
	if (param.getOrientMethode()==1 && param.getBascOnPlan() && param.getMasque()!=QString()) {
		xmlWriter.writeStartElement(QString("BasculeOrientation"));
			xmlWriter.writeTextElement(QString("PatternNameEstim"), param.getMasque());
			xmlWriter.writeStartElement(QString("ModeBascule"));
				xmlWriter.writeStartElement(QString("BasculeLiaisonOnPlan"));
					xmlWriter.writeStartElement(QString("EstimPl"));
						xmlWriter.writeTextElement(QString("AttrSup"), QString("Plan"));
						xmlWriter.writeTextElement(QString("KeyCalculMasq"), QString("Key-Assoc-Std-MultiMasq-Image"));
						xmlWriter.writeTextElement(QString("IdBdl"), QString("Id_Pastis_Hom"));
						xmlWriter.writeStartElement(QString("Pond"));
							xmlWriter.writeTextElement(QString("EcartMesureIndiv"), QString("1.0"));
							xmlWriter.writeTextElement(QString("Show"), QString("eNSM_Paquet"));
							xmlWriter.writeTextElement(QString("NbMax"), QString("100"));
							xmlWriter.writeTextElement(QString("ModePonderation"), QString("eL1Secured"));
							xmlWriter.writeTextElement(QString("SigmaPond"), QString("2.0"));
							xmlWriter.writeTextElement(QString("EcartMax"), QString("15.0"));
						xmlWriter.writeEndElement();	//Pond
					xmlWriter.writeEndElement();	//EstimPl
				xmlWriter.writeEndElement();	//BasculeLiaisonOnPlan
			xmlWriter.writeEndElement();	//ModeBascule
		xmlWriter.writeEndElement();	//BasculeOrientation
	}

	//orientation dans le plan
	if (param.getOrientMethode()==1 && param.getBascOnPlan() && !param.getImage1().isEmpty() && !param.getImage2().isEmpty() && param.getPoint1()!=QPoint(-1,-1) && param.getPoint2()!=QPoint(-1,-1)) {
		xmlWriter.writeStartElement(QString("FixeOrientPlane"));
			xmlWriter.writeStartElement(QString("ModeFOP"));
				xmlWriter.writeStartElement(QString("HorFOP"));
					xmlWriter.writeStartElement(QString("VecFOH"));
						QPoint P1 = param.getPoint1();
						xmlWriter.writeTextElement(QString("Pt"), QVariant(P1.x()).toString()+QString(" ")+QVariant(P1.y()).toString());
						xmlWriter.writeTextElement(QString("Im"), param.getImage1());
					xmlWriter.writeEndElement();	//VecFOH
					xmlWriter.writeStartElement(QString("VecFOH"));
						QPoint P2 = param.getPoint2();
						xmlWriter.writeTextElement(QString("Pt"), QVariant(P2.x()).toString()+QString(" ")+QVariant(P2.y()).toString());
						xmlWriter.writeTextElement(QString("Im"), param.getImage2());
					xmlWriter.writeEndElement();	//VecFOH
				xmlWriter.writeEndElement();	//HorFOP
			xmlWriter.writeEndElement();	//ModeFOP
			xmlWriter.writeTextElement(QString("Vecteur"), QString("1 0"));
		xmlWriter.writeEndElement();	//FixeOrientPlane
	}

	//mise à l'échelle échelle
	if ((param.getOrientMethode()==1 && param.getFixEchelle()) || param.getOrientMethode()==2) {
		bool b = true;
		if (param.getDistance()==0) b = false;
		else {
			for (int i=0; i<4; i++) {
				if (param.getPoints().at(i)==QPoint(-1,-1) || param.getImages().at(i).isEmpty()) {
					b = false;
					break;
				}
			}
		}
		if (b) {
			xmlWriter.writeStartElement(QString("FixeEchelle"));
				xmlWriter.writeStartElement(QString("ModeFE"));
					xmlWriter.writeStartElement(QString("StereoFE"));
						for (int i=0; i<2; i++) {	//point stéréo
							xmlWriter.writeStartElement(QString("HomFE"));
							for (int j=0; j<2; j++) {	//vue
								QPoint P = param.getPoints().at(2*j+i);
								xmlWriter.writeTextElement(QString("P%1").arg(j+1), QString("%1 %2").arg(P.x()).arg(P.y()));
								xmlWriter.writeTextElement(QString("Im%1").arg(j+1), param.getImages().at(2*j+i));
							}
							xmlWriter.writeEndElement();	//HomFE
						}
					xmlWriter.writeEndElement();	//StereoFE
				xmlWriter.writeEndElement();	//ModeFE
				xmlWriter.writeTextElement(QString("DistVraie"), QVariant(param.getDistance()).toString());
			xmlWriter.writeEndElement();	//FixeEchelle
		}
	}

	//géoréférencement
	if (param.getOrientMethode()==3 || param.getOrientMethode()==4) {
		QString imagesPattern("(");
		if (param.getOrientMethode()==4) {
			for (int i=0; i<images.count(); i++) {
				imagesPattern += images.at(i);
				if (i<images.count()-1) imagesPattern += QString("|");
			}
			imagesPattern += QString(")");
		}

		xmlWriter.writeStartElement(QString("BasculeOrientation"));
			//xmlWriter.writeTextElement(QString("AfterCompens"), QString("false"));
			if (param.getOrientMethode()==3) xmlWriter.writeTextElement(QString("PatternNameEstim"), QString(".*"));
			else xmlWriter.writeTextElement(QString("PatternNameEstim"), imagesPattern);
			xmlWriter.writeStartElement(QString("ModeBascule"));
				xmlWriter.writeStartElement(QString("BasculeOnPoints"));
					if (param.getOrientMethode()==3) {
						xmlWriter.writeStartElement(QString("BascOnAppuis"));
							xmlWriter.writeTextElement(QString("NameRef"), QString("Id-Appui"));
						xmlWriter.writeEndElement();	//BascOnAppuis
					} else {
						xmlWriter.writeTextElement(QString("BascOnCentre"), QString(""));
					}
					xmlWriter.writeTextElement(QString("ModeL2"), QString("true"));
				xmlWriter.writeEndElement();	//BasculeOnPoints
			xmlWriter.writeEndElement();	//ModeBascule
		xmlWriter.writeEndElement();	//BasculeOrientation
	}

	xmlWriter.writeEndDocument();
	return true;
}

QString FichierBasculOri::lire (const QString& fichier, UserOrientation& param, const ParamMain& paramMain) {
	if (param.getOrientMethode()==0 || param.getOrientMethode()==3) return QString();
	QString sbase = conv(QObject::tr("Reorientation file %1 reading :\n")).arg(fichier);
	XmlTree xmlTree(fichier);
	QString err = xmlTree.lire(false);
	if (!err.isEmpty()) return err;

	bool ok;
	XmlTag tag0;
	XmlTag& tag = *(&tag0);

	const XmlTag* basculeTag(0), * fixOriTag(0), * fixEchTag(0);
	if (param.getOrientMethode()!=2 && param.getOrientMethode()!=3 && xmlTree.getMainTag().getNom()==QString("BasculeOrientation")) {
		basculeTag = &(xmlTree.getMainTag());
		if (param.getOrientMethode()==1) {
			param.setBascOnPlan(true);
			for (int i=0; i<xmlTree.getOtherTags().count(); i++) {
				if (xmlTree.getOtherTags().at(i).getNom()==QString("FixeOrientPlane")) fixOriTag = &(xmlTree.getOtherTags().at(i));
				else if (xmlTree.getOtherTags().at(i).getNom()==QString("FixeEchelle")) fixEchTag = &(xmlTree.getOtherTags().at(i));
			}
		}
	} else if (param.getOrientMethode()!=3) {
		if (param.getOrientMethode()==4) return sbase+conv(QObject::tr("This file is not a reorientation file."));
		else if ((param.getOrientMethode()==2 || param.getOrientMethode()==1) && xmlTree.getMainTag().getNom()==QString("FixeEchelle")) {
			fixEchTag = &(xmlTree.getMainTag());
			if (param.getOrientMethode()==1) param.setBascOnPlan(false);
		} else return sbase+conv(QObject::tr("This file is not a reorientation file."));
	}

	if (basculeTag) {
		tag = basculeTag->getEnfant("PatternNameEstim",&ok);
		if (!ok) return sbase+conv(QObject::tr("Images to reorientate not found."));
		QString text = tag.getContenu();
		if (param.getOrientMethode()==1) {
			param.setImgMasque(text);
		} else if (param.getOrientMethode()==4) {



		}
	}

	if (fixOriTag) {
		//segment
		const XmlTag* modeTag = &(fixOriTag->getEnfant("ModeFOP",&ok));
		if (!ok) return sbase+conv(QObject::tr("Parameters to fix orientation not found."));
		modeTag = &(fixOriTag->getEnfant("HorFOP",&ok));
		if (!ok) return sbase+conv(QObject::tr("Parameters to fix orientation not found."));
		const QList<const XmlTag*> ptsListe = modeTag->getEnfants("VecFOH",&ok);
		if (!ok) return sbase+conv(QObject::tr("No points to fix orientation found."));
		QPoint P1, P2;
		QString img1, img2;
		for (int i=0; i<2; i++) {
			tag = ptsListe.at(i)->getEnfant("Pt",&ok);
			if (!ok) return sbase+conv(QObject::tr("A point to fix orientation has no coordinates."));
			QString text = tag.getContenu();
			int x = text.section(" ",0,0).toInt(&ok);
			if (!ok) return sbase+conv(QObject::tr("A coordinate of a point to fix orientation is unvalid."));
			int y = text.section(" ",1,1).toInt(&ok);
			if (!ok) return sbase+conv(QObject::tr("A coordinate of a point to fix orientation is unvalid."));
			if (i==0) P1 = QPoint(x,y);
			else P2 = QPoint(x,y);

			tag = ptsListe.at(i)->getEnfant("Im",&ok);
			if (!ok) return sbase+conv(QObject::tr("A coordinate of a GCP measure to fix orientation is unvalid."));
			if (i==0) img1 = tag.getContenu();
			else img2 = tag.getContenu();
		}

		//direction
		tag = fixOriTag->getEnfant("Vecteur",&ok);
		if (!ok) return sbase+conv(QObject::tr("Direction vector to fix orientation not found."));
		QString text = tag.getContenu();
		int x = text.section(" ",0,0).toInt(&ok);
		if (!ok) return sbase+conv(QObject::tr("A coordinate of the axis to fix orientation is unvalid."));
		int y = text.section(" ",1,1).toInt(&ok);
		if (!ok) return sbase+conv(QObject::tr("A coordinate of the axis to fix orientation is unvalid."));
		
		P2 = P2-P1;
		x /= sqrt( (double)( x*x+y*y ) );
		y /= sqrt( (double)( x*x+y*y ) );
		int x2b = P2.x()*x + P2.y()*y;
		int y2b = -P2.x()*y + P2.y()*x;
		P2 = QPoint(x2b,y2b);
		P2 = P2+P1;

		param.setPoint1(P1);
		param.setPoint2(P2);
		param.setImage1(img1);
		param.setImage2(img2);
	}
	
	if (fixEchTag) {
		param.setFixEchelle(true);
		const XmlTag* modeTag = &(fixOriTag->getEnfant("ModeFE",&ok));
		if (!ok) return sbase+conv(QObject::tr("Parameters to fix scale not found."));
		modeTag = &(fixOriTag->getEnfant("StereoFE",&ok));
		if (!ok) return sbase+conv(QObject::tr("Parameters to fix scale not found."));
		const QList<const XmlTag*> homListe = modeTag->getEnfants("HomFE",&ok);
		if (!ok) return sbase+conv(QObject::tr("No tie-points to fix scale were found."));
		param.modifPoints().resize(4);
		param.modifImages().resize(4);
		for (int i=0; i<2; i++) {
			for (int j=0; j<2; j++) {
				tag = homListe.at(i)->getEnfant(QString("P%1").arg(j+1),&ok);
				if (!ok) return sbase+conv(QObject::tr("A tie-point to fix scale was not found."));
				QString text = tag.getContenu();
				int x = text.section(" ",0,0).toInt(&ok);
				if (!ok) return sbase+conv(QObject::tr("A coordinate of a tie-point to fix scale is unvalid."));
				int y = text.section(" ",1,1).toInt(&ok);
				if (!ok) return sbase+conv(QObject::tr("A coordinate of a tie-point to fix scale is unvalid."));
				param.modifPoints()[2*j+i] = QPoint(x,y);

				tag = homListe.at(i)->getEnfant(QString("Im%1").arg(j+1),&ok);
				if (!ok) return sbase+conv(QObject::tr("An image to fix scale was not found."));
				param.modifImages()[2*j+i] = tag.getContenu();			
			}
		}

		tag = fixEchTag->getEnfant("DistVraie",&ok);
		if (!ok) return sbase+conv(QObject::tr("Ground distance to fix scale was not found."));
		double d = tag.getContenu().toDouble(&ok);
		if (!ok) return sbase+conv(QObject::tr("Ground distance to fix scale is unvalid."));
		param.setDistance(d);
	} else
		param.setFixEchelle(false);

	if (param.getOrientMethode()==3) {
		//points topo
		QString fichier2 = paramMain.getDossier()+paramMain.getDefIncGPSXML();
		sbase = conv(QObject::tr("GCP definition file %1 reading :\n")).arg(fichier2);
		XmlTree xmlTree2(fichier2);
		err = xmlTree2.lire();
		if (!err.isEmpty()) return err;

		if (xmlTree2.getMainTag().getNom()!=QString("PointFlottantInc")) return sbase+conv(QObject::tr("This file is not a GCP definition file."));
		tag = xmlTree2.getMainTag().getEnfant("KeySetOrPat",&ok);
		if (!ok) return sbase+conv(QObject::tr("GCP file not found."));
		param.setPointsGPS(tag.getContenu().section("^",-1,-1));

		//mesures image
		QString fichier3 = paramMain.getDossier()+paramMain.getDefObsGPSXML();
		sbase = conv(QObject::tr("Image measure definition file %1 reading :\n")).arg(fichier3);
		XmlTree xmlTree3(fichier3);
		err = xmlTree3.lire();
		if (!err.isEmpty()) return err;

		if (xmlTree3.getMainTag().getNom()!=QString("BDD_ObsAppuisFlottant")) return sbase+conv(QObject::tr("This file is not an image measure definition file."));
		tag = xmlTree2.getMainTag().getEnfant("KeySetOrPat",&ok);
		if (!ok) return sbase+conv(QObject::tr("Image measure file not found."));
		param.setAppuisImg(tag.getContenu().section("^",-1,-1));
	}
	if (param.getOrientMethode()==4) {

	}

	return QString();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//fichier de points d'appui terrain
//le fichier texte doit contenir les coordonnées comme suit :
//	nomPoint X Y Z dx dy dz (tout en mètres)
FichierAppuiGPS::FichierAppuiGPS() {}

bool FichierAppuiGPS::convert (const QString& fichierold, const QString& fichiernew) {
	//convert le fichier du format texte au format xml s'il y a lieu
	if (QFile(fichiernew).exists()) QFile(fichiernew).remove();
	bool texte = true;
	if (!format(fichierold,texte)) return false;
	if (!texte) {
		QFile(fichierold).copy(fichiernew);
		return true;
	}

	QFile oldFile(fichierold);
	if (!oldFile.open(QFile::ReadOnly | QFile::Text))
		return false;
	QList<QString> points;
	QList<QVector<double> > coordonnees;
	QString err = lireFichierTexte(oldFile, points, coordonnees);
	if (!err.isEmpty()) {
		cout << err.toStdString() << endl;
		return false;
	}

	QFile newFile(fichiernew);
	if (!newFile.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate))
		return false;
	QXmlStreamWriter xmlWriter(&newFile);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	xmlWriter.writeStartElement(QString("DicoAppuisFlottant"));
	for (int i=0; i<points.count(); i++) {		
		xmlWriter.writeStartElement(QString("OneAppuisDAF"));
			xmlWriter.writeTextElement(QString("Pt"), QString("%1 %2 %3").arg(coordonnees.at(i).at(0)).arg(coordonnees.at(i).at(1)).arg(coordonnees.at(i).at(2)));
			xmlWriter.writeTextElement(QString("NamePt"),points.at(i));
			xmlWriter.writeTextElement(QString("Incertitude"), QString("%1 %2 %3").arg(coordonnees.at(i).at(3)).arg(coordonnees.at(i).at(4)).arg(coordonnees.at(i).at(5)));
		xmlWriter.writeEndElement();	//OneAppuisDAF
	}

	xmlWriter.writeEndElement();	//DicoAppuisFlottant
	xmlWriter.writeEndDocument();
	newFile.close();
	return true;
}

bool FichierAppuiGPS::format(const QString& fichier, bool& texte) {
	//vérifie si le fichier à convertir est au format texte (true) ou déjà au format xml (false)
	QFile file(fichier);
	if (!file.open(QFile::ReadOnly | QFile::Text))
		return false;
	QTextStream textReader(&file);
	QString s = textReader.readAll();
	texte = !(s.contains("<DicoAppuisFlottant>"));
	file.close();
	return true;
}

QString FichierAppuiGPS::lire(const QString& fichier, QList<QString>& points) {
	QString sbase = conv(QObject::tr("GCP file %1 reading :\n")).arg(fichier);
	QFile file(fichier);
	if (!file.open(QFile::ReadOnly | QFile::Text)) {
			return sbase+conv(QObject::tr("Fail to open file."));
	}
	points.clear();
	bool texte = true;
	if (!format(fichier,texte)) return sbase+conv(QObject::tr("Fail to determine file format."));

	if (!texte) {
		XmlTree xmlTree(fichier);
		QString err = xmlTree.lire();
		if (!err.isEmpty()) return err;

		bool ok;
		XmlTag tag0;
		XmlTag& tag = *(&tag0);
		if (xmlTree.getMainTag().getNom()!=QString("DicoAppuisFlottant")) {
			if (xmlTree.getMainTag().getNom()!=QString("Global")) return sbase+conv(QObject::tr("This file is not in Apero format."));
			tag = xmlTree.getMainTag().getEnfant("DicoAppuisFlottant",&ok);
			if (!ok) return sbase+conv(QObject::tr("This file is not a GCP list."));
		} else
			tag = xmlTree.getMainTag();

		const QList<const XmlTag*> ptsListe = tag.getEnfants("OneAppuisDAF",&ok);
		if (!ok) return sbase+conv(QObject::tr("No points found."));
		for (QList<const XmlTag*>::const_iterator it=ptsListe.begin(); it!=ptsListe.end(); it++) {
			tag = (*it)->getEnfant("NamePt",&ok);
			if (!ok) return sbase+conv(QObject::tr("A point has no name."));
			points.push_back(tag.getContenu());
		}	
	} else {
		QFile file(fichier);
		if (!file.open(QFile::ReadOnly | QFile::Text))
			return sbase+conv(QObject::tr("Fail to open file."));
		QList<QVector<double> > coordonnees;
		QString err = lireFichierTexte(file, points, coordonnees);
		if (!err.isEmpty()) return sbase+err;	
	}
	return QString();
}

QString FichierAppuiGPS::lireFichierTexte(QFile& fichier, QList<QString>& points, QList<QVector<double> >& coordonnees) {
	QString v,p;
	systemeNumerique(v,p);
	QTextStream textReader(&fichier);
	while (!textReader.atEnd()) {
		QString point = textReader.readLine().trimmed().simplified();
		QString station = point.section(" ",0,0);
		bool b = false;
		double X = point.section(" ",1,1).replace(p,v).toDouble(&b);
		if (!b) {
			return conv(QObject::tr("%1 : unvalid coordinate\n").arg(point.section(" ",1,1)));
		}
		double Y = point.section(" ",2,2).replace(p,v).toDouble(&b);
		if (!b) {
			return conv(QObject::tr("%1 : unvalid coordinate\n").arg(point.section(" ",2,2)));
		}
		double Z = point.section(" ",3,3).replace(p,v).toDouble(&b);
		if (!b) {
			return conv(QObject::tr("%1 : unvalid coordinate\n").arg(point.section(" ",3,3)));
		}

		double dx = point.section(" ",4,4).replace(p,v).toDouble(&b);
		if (!b) {
			return conv(QObject::tr("%1 : this precision is not a number\n").arg(point.section(" ",4,4)));
		}
		double dy = point.section(" ",4,4).replace(p,v).toDouble(&b);
		if (!b) {
			return conv(QObject::tr("%1 : this precision is not a number\n").arg(point.section(" ",4,4)));
		}
		double dz = point.section(" ",5,5).replace(p,v).toDouble(&b);
		if (!b) {
			return conv(QObject::tr("%1 : this precision is not a number\n").arg(point.section(" ",5,5)));
		}
		points.push_back(station);
		QVector<double> c;
		c << X << Y << Z << dx << dy << dz;
		coordonnees.push_back(c);
	}
	fichier.close();
	return QString();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//fichier de points d'appui avec leurs coordonnées dans les images
//le fichier texte doit contenir les coordonnées comme suit :
//	nomImage colonne ligne nomPointTerrain (en pixels)
FichierAppuiImage::FichierAppuiImage() {}

bool FichierAppuiImage::convert (const QString& fichierold, const QString& fichiernew) {
	//convert le fichier du format texte au format xml s'il y a lieu
	if (QFile(fichiernew).exists()) QFile(fichiernew).remove();
	bool texte = true;
	if (!format(fichierold,texte)) return false;
	if (!texte) {
		QFile(fichierold).copy(fichiernew);
		return true;
	}

	//lecture
	QFile oldFile(fichierold);
	if (!oldFile.open(QFile::ReadOnly | QFile::Text))
		return false;
	QTextStream textReader(&oldFile);

	QString v,p;
	systemeNumerique(v,p);
	QList<pair<QString, QList<pair<QString,QPoint> > > > donnees;	//image , liste <station,coord>

	while (!textReader.atEnd()) {
		QString mesure = textReader.readLine().trimmed().simplified();
		QString image = mesure.section(" ",0,0);
		QString station = mesure.section(" ",3,3);
		bool b = false;
		double x = mesure.section(" ",1,1).replace(p,v).toDouble(&b);
		if (!b) {
			cout << QObject::tr("%1 : unvalid coordinate\n").arg(mesure.section(" ",1,1)).toStdString();
			return false;
		}
		double y = mesure.section(" ",2,2).replace(p,v).toDouble(&b);
		if (!b) {
			cout << QObject::tr("%1 : unvalid coordinate\n").arg(mesure.section(" ",2,2)).toStdString();
			return false;
		}
		
		//enregistrement
		b = false;
		for (QList<pair<QString, QList<pair<QString,QPoint> > > >::iterator it=donnees.begin(); it!=donnees.end(); it++) {
			if (it->first!=image) continue;
			it->second.push_back( pair<QString,QPoint>(station, QPoint(x,y)) );
			b = true;
			break;
		}
		if (!b) {
			donnees.push_back( pair<QString, QList<pair<QString,QPoint> > >( image, QList<pair<QString,QPoint> >() ) );
			donnees.last().second.push_back( pair<QString,QPoint>(station, QPoint(x,y)) );
		}
	}

	oldFile.close();

	//écriture
	QFile newFile(fichiernew);
	if (!newFile.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate))
		return false;
	QXmlStreamWriter xmlWriter(&newFile);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	xmlWriter.writeStartElement(QString("SetOfMesureAppuisFlottants"));

	for (QList<pair<QString, QList<pair<QString,QPoint> > > >::iterator it=donnees.begin(); it!=donnees.end(); it++) {
		xmlWriter.writeStartElement(QString("MesureAppuiFlottant1Im"));
			xmlWriter.writeTextElement(QString("NameIm"), it->first);

			for (QList<pair<QString,QPoint> >::iterator it2=it->second.begin(); it2!=it->second.end(); it2++) {
				xmlWriter.writeStartElement(QString("OneMesureAF1I"));
					xmlWriter.writeTextElement(QString("NamePt"), it2->first);
					xmlWriter.writeTextElement(QString("PtIm"), QString("%1 %2").arg(it2->second.x()).arg(it2->second.y()));
				xmlWriter.writeEndElement();	//OneMesureAF1I
			}
		xmlWriter.writeEndElement();	//MesureAppuiFlottant1Im
	}

	xmlWriter.writeEndElement();	//SetOfMesureAppuisFlottants
	xmlWriter.writeEndDocument();
	newFile.close();

	return true;
}

bool FichierAppuiImage::format(const QString& fichier, bool& texte) {
	//vérifie si le fichier à convertir est au format texte (true) ou déjà au format xml (false)
	QFile file(fichier);
	if (!file.open(QFile::ReadOnly | QFile::Text))
		return false;
	QTextStream textReader(&file);
	QString s = textReader.readAll();
	texte = !(s.contains("<SetOfMesureAppuisFlottants>"));
	file.close();
	return true;
}

QString FichierAppuiImage::lire(const QString& fichier, const ParamMain* paramMain, const QList<QString>& points, QVector<QVector<QPoint> >& pointsAppui) {
	QString p,v;
	systemeNumerique(v,p);
	QString sbase = conv(QObject::tr("GCP measure file %1 reading :\n")).arg(fichier);
	QFile file(fichier);
	if (!file.open(QFile::ReadOnly | QFile::Text)) {
			return sbase+conv(QObject::tr("Fail to open file."));
	}
	bool texte = true;
	if (!format(fichier,texte)) return sbase+conv(QObject::tr("Fail to determine file format."));

	if (!texte) {
		XmlTree xmlTree(fichier);
		QString err = xmlTree.lire();
		if (!err.isEmpty()) return err;

		bool ok;
		XmlTag tag0;
		XmlTag& tag = *(&tag0);
		if (xmlTree.getMainTag().getNom()!=QString("SetOfMesureAppuisFlottants")) return sbase+conv(QObject::tr("This file is not in Apero format."));

		const QList<const XmlTag*> mesListe = xmlTree.getMainTag().getEnfants("MesureAppuiFlottant1Im",&ok);
		if (!ok) return sbase+conv(QObject::tr("No measures found."));
		for (QList<const XmlTag*>::const_iterator it=mesListe.begin(); it!=mesListe.end(); it++) {
				//image
			tag = (*it)->getEnfant("NameIm",&ok);
			if (!ok) return sbase+conv(QObject::tr("The image name of a measure is missing."));
			QString image = tag.getContenu();
			int idxImg = paramMain->findImg(image, 1);
			if (idxImg==-1) idxImg = paramMain->findImg(image, 0);
			if (idxImg==-1) return sbase+conv(QObject::tr("Image %1 does not match any survey image.\n").arg(image));

				//point
			const QList<const XmlTag*> mesListe2 = (*it)->getEnfants("OneMesureAF1I",&ok);
			for (QList<const XmlTag*>::const_iterator it2=mesListe2.begin(); it2!=mesListe2.end(); it2++) {
				tag = (*it2)->getEnfant("NamePt",&ok);
				if (!ok) return sbase+conv(QObject::tr("The GCP name of a measure is missing."));
				QString station = tag.getContenu();
				int idxGPS = points.indexOf(station);
				if (idxGPS==-1) return sbase+conv(QObject::tr("GCP %1 does not correspond to any point from the GCP file.\n").arg(station));
				
				//coordonnées
				tag = (*it2)->getEnfant("PtIm",&ok);
				if (!ok) return sbase+conv(QObject::tr("The image coordinates of a measure are missing."));
				QString coord = tag.getContenu();
				coord.replace(p,v);
				int x = coord.section(" ",0,0).toInt(&ok);
				if (!ok) return sbase+conv(QObject::tr("%1 : this coordinate is not a number").arg(coord.section(" ",0,0)));
				int y = coord.section(" ",1,1).toInt(&ok);
				if (!ok) return sbase+conv(QObject::tr("%1 : this coordinate is not a number").arg(coord.section(" ",1,1)));
				pointsAppui[idxImg][idxGPS] = QPoint(x,y);
			}
		}
	} else {
		QList<QVector<double> > coordonnees;
		QFile file(fichier);
		if (!file.open(QFile::ReadOnly | QFile::Text))
			return sbase+conv(QObject::tr("Fail to open file."));
		QString err = lireFichierTexte(file, paramMain, points, pointsAppui);
		if (!err.isEmpty()) return sbase+err;	
	}
	return QString();
}

QString FichierAppuiImage::lireFichierTexte(QFile& fichier, const ParamMain* paramMain, const QList<QString>& points, QVector<QVector<QPoint> >& pointsAppui) {
	QString p,v;
	systemeNumerique(v,p);
	QTextStream textReader(&fichier);
	while (!textReader.atEnd()) {
		//lecture
		QString point = textReader.readLine().trimmed().simplified();
		QString image = point.section(" ",0,0);
		bool b = false;
		int x = point.section(" ",1,1).replace(p,v).toInt(&b);
		if (!b) {
			return conv(QObject::tr("%1 : this coordinate is not a number").arg(point.section(" ",1,1)));
		}
		int y = point.section(" ",2,2).replace(p,v).toInt(&b);
		if (!b) {
			return conv(QObject::tr("%1 : this coordinate is not a number").arg(point.section(" ",2,2)));
		}
		QString station = point.section(" ",3,3);

		//enregistrement
			//image
		int idxImg = paramMain->findImg(image, 1);
		if (idxImg==-1) idxImg = paramMain->findImg(image, 0);
		if (idxImg==-1)
			return conv(QObject::tr("Image %1 does not match any survey image.\n").arg(image));
			//point GPS
		int idxGPS = points.indexOf(station);
		if (idxGPS==-1)
			return conv(QObject::tr("GCP %1 does not match an GCP extracted from file.\n").arg(station));
			//coordonnées
		pointsAppui[idxImg][idxGPS] = QPoint(x,y);
	}
	fichier.close();
	return QString();
}

bool FichierAppuiImage::ecrire(const QString& fichier, const ParamMain* paramMain, const QList<QString>& points, const QVector<QVector<QPoint> >& pointsAppui) {
	if (QFile(fichier).exists()) QFile(fichier).remove();
	QFile file(fichier);
	if (!file.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) return false;
	QXmlStreamWriter xmlWriter(&file);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	xmlWriter.writeStartElement(QString("SetOfMesureAppuisFlottants"));
	for (int i=0; i<points.count(); i++) {	
	for (int j=0; j<pointsAppui.count(); j++) {
		if (pointsAppui.at(j).at(i)==QPoint(-1,-1)) continue;
		xmlWriter.writeStartElement(QString("MesureAppuiFlottant1Im"));
			xmlWriter.writeTextElement(QString("NameIm"), paramMain->getCorrespImgCalib().at(j).getImageTif());
			xmlWriter.writeStartElement(QString("OneMesureAF1I"));
				xmlWriter.writeTextElement(QString("NamePt"), points.at(i));
				xmlWriter.writeTextElement(QString("PtIm"), QString("%1 %2").arg(pointsAppui.at(j).at(i).x()).arg(pointsAppui.at(j).at(i).y()));
			xmlWriter.writeEndElement();	//OneMesureAF1I
		xmlWriter.writeEndElement();	//MesureAppuiFlottant1Im
	}}

	xmlWriter.writeEndElement();	//SetOfMesureAppuisFlottants
	xmlWriter.writeEndDocument();
	file.close();
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//fichier pointant vers le fichier des mesures image des points GPS
FichierObsGPS::FichierObsGPS() {}

bool FichierObsGPS::ecrire (const QString& fichier, const UserOrientation& param) {
	QFile file(fichier);
	if (!file.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate))
		return false;
	QXmlStreamWriter xmlWriter(&file);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	if (param.getOrientMethode()==3) {
		xmlWriter.writeStartElement(QString("BDD_ObsAppuisFlottant"));
			xmlWriter.writeTextElement(QString("Id"), QString("Id-Appui"));
			xmlWriter.writeTextElement(QString("KeySetOrPat"), QString("^")+param.getAppuisImg().section("/",-1,-1));
		xmlWriter.writeEndElement();	//BDD_ObsAppuisFlottant1Im
	} else if (param.getOrientMethode()==4) {
		xmlWriter.writeStartElement(QString("BDD_Centre"));
			xmlWriter.writeTextElement(QString("Id"), QString("Id-Centre"));
			xmlWriter.writeTextElement(QString("KeySet"), QString("NKS-Set-Orient@-BDDC"));
			xmlWriter.writeTextElement(QString("KeyAssoc"), QString("NKS-Assoc-Im2Orient@-BDDC"));
		xmlWriter.writeEndElement();	//BDD_Centre
	}

	xmlWriter.writeEndDocument();
	file.close();
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//fichier pointant vers le fichier des points GPS
FichierIncGPS::FichierIncGPS() {}

bool FichierIncGPS::ecrire (const QString& fichier, const UserOrientation& param) {
	QFile file(fichier);
	if (!file.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate))
		return false;
	QXmlStreamWriter xmlWriter(&file);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	xmlWriter.writeStartElement(QString("PointFlottantInc"));
		xmlWriter.writeTextElement(QString("Id"), QString("Id-Appui"));
		xmlWriter.writeTextElement(QString("KeySetOrPat"), QString("^")+param.getPointsGPS().section("/",-1,-1));
	xmlWriter.writeEndElement();	//PointFlottantInc1Im

	xmlWriter.writeEndDocument();
	file.close();
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//fichier pour la pondération des points GPS
FichierPondGPS::FichierPondGPS() {}

bool FichierPondGPS::ecrire (const QString& fichier, int orientMethode) {
	QFile file(fichier);
	if (!file.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate))
		return false;
	QXmlStreamWriter xmlWriter(&file);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	if (orientMethode==3) {
		xmlWriter.writeStartElement(QString("ObsAppuisFlottant"));	
			xmlWriter.writeTextElement(QString("NameRef"), QString("Id-Appui"));
			xmlWriter.writeStartElement(QString("PondIm"));
				xmlWriter.writeTextElement(QString("EcartMesureIndiv"), QString("1.0"));
				xmlWriter.writeTextElement(QString("Show"), QString("eNSM_Paquet"));
				xmlWriter.writeTextElement(QString("NbMax"), QString("100"));
				xmlWriter.writeTextElement(QString("ModePonderation"), QString("eL1Secured"));
				xmlWriter.writeTextElement(QString("SigmaPond"), QString("20.0"));
				xmlWriter.writeTextElement(QString("EcartMax"), QString("5000000.0"));
			xmlWriter.writeEndElement();	//PondIm
			xmlWriter.writeTextElement(QString("PtsShowDet"), QString(".*"));
			xmlWriter.writeTextElement(QString("DetShow3D"), QString("0"));
			xmlWriter.writeTextElement(QString("ShowMax"), QString("true"));
			xmlWriter.writeTextElement(QString("ShowSom"), QString("true"));
		xmlWriter.writeEndElement();	//ObsAppuisFlottant
	} else {
		xmlWriter.writeStartElement(QString("ObsCentrePDV"));	
			xmlWriter.writeTextElement(QString("PatternApply"), QString(".*"));
			xmlWriter.writeStartElement(QString("Pond"));
				xmlWriter.writeTextElement(QString("EcartMesureIndiv"), QString("1.0"));
				xmlWriter.writeTextElement(QString("Show"), QString("eNSM_Paquet"));
				xmlWriter.writeTextElement(QString("ModePonderation"), QString("eL1Secured"));
			xmlWriter.writeEndElement();	//Pond
		xmlWriter.writeEndElement();	//ObsCentrePDV
	}

	xmlWriter.writeEndDocument();
	file.close();
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//fichier de coordonnées GPS de sommets
//le fichier texte doit contenir les coordonnées comme suit :
//	nomImage X Y Z
FichierSommetsGPS::FichierSommetsGPS() {}

bool FichierSommetsGPS::convert (const QString& fichierold, const ParamMain& paramMain, QString& resultDir, QStringList& images) {
	//convert le fichier fichierold du format texte en un dossier resultDir au format xml s'il y a lieu ; images est la liste des images à basculer
	if (QDir(fichierold).exists()) {
		resultDir = fichierold;	
		//images
		QStringList l = QDir(resultDir).entryList(QDir::Files);
		for (QStringList::const_iterator it=l.begin(); it!=l.end(); it++) {
			if (it->left(1)==QString(".")) continue;
			QString image = *it;
			if (image.left(12)!=QString("Orientation-")) {
				int pos = it->indexOf("-F",0);
				image = QString("Orientation-")+image.right(image.count()-pos-1);
				QFile(resultDir+*it).rename(resultDir+image);
			}
			image = image.right(image.count()-12).section(".",0,-2);
			if (paramMain.findImg(image,1)==-1) {
				cout << conv(QObject::tr("Image %1 does not match any survey image.")).arg(image).toStdString() << endl;
				return false;
			} else
				images.push_back(image);
		}
		return true;
	} else {
		QString v,p;
		systemeNumerique(v,p);

		//nouveau dossier
		resultDir = paramMain.getDossier()+QString("Ori-BDDC/");
		if (QDir(resultDir).exists()) {
			int i = 0;
			while (QDir(paramMain.getDossier()+QString("Ori-BDDC%1").arg(i)).exists()) i++;
			QDir(paramMain.getDossier()).rename(QString("Ori-BDDC"),QString("Ori-BDDC%1").arg(i));
		}
		if (!QDir(paramMain.getDossier()).mkdir("Ori-BDDC")) {
			cout << conv(QObject::tr("Fail to create directory %1.")).arg(resultDir).toStdString() << endl;	
			return false;
		}

		//conversion
		QFile oldFile(fichierold);
		if (!oldFile.open(QFile::ReadOnly | QFile::Text)) {
			cout << conv(QObject::tr("Fail to open file %1.")).arg(fichierold).toStdString() << endl;
			return false;
		}
		QTextStream textReader(&oldFile);

		images.clear();
		while (!textReader.atEnd()) {
			QString sommet = textReader.readLine().trimmed().simplified();
			if (sommet.isEmpty()) continue;
			QString image = sommet.section(" ",0,0);
			int idx = paramMain.findImg(image,1,false);
			if (idx==-1) idx = paramMain.findImg(image,0);
			if (idx==-1) {
				cout << conv(QObject::tr("Image %1 does not match any survey image.")).arg(image).toStdString() << endl;
				return false;
			}
			image = paramMain.getCorrespImgCalib().at(idx).getImageTif();
			images.push_back(image);

			bool b = false;
			double x = sommet.section(" ",1,1).replace(p,v).toDouble(&b);
			if (!b) {
				cout << QObject::tr("%1 : this coordinate is not a number").arg(sommet.section(" ",1,1)).toStdString() << endl;;
				return false;
			}
			double y = sommet.section(" ",2,2).replace(p,v).toDouble(&b);
			if (!b) {
				cout << QObject::tr("%1 : this coordinate is not a number").arg(sommet.section(" ",2,2)).toStdString() << endl;
				return false;
			}
			double z = sommet.section(" ",3,3).replace(p,v).toDouble(&b);
			if (!b) {
				cout << QObject::tr("%1 : this coordinate is not a number").arg(sommet.section(" ",2,2)).toStdString() << endl;
				return false;
			}

			bool doRot = (!sommet.section(" ",4,4).isEmpty() && !sommet.section(" ",5,5).isEmpty() && !sommet.section(" ",6,6).isEmpty());
			double roll = 0.,
				   pitch = 0.,
				   direction = 0.;
			if (doRot) {
				roll = sommet.section(" ",4,4).replace(p,v).toDouble(&b);
				if (!b) {
					cout << ch(QObject::tr("%1 : this coordinate is not a number").arg(sommet.section(" ",4,4))) << endl;;
					return false;
				}
				pitch = sommet.section(" ",5,5).replace(p,v).toDouble(&b);
				if (!b) {
					cout << QObject::tr("%1 : this coordinate is not a number").arg(sommet.section(" ",5,5)).toStdString() << endl;
					return false;
				}
				direction = sommet.section(" ",6,6).replace(p,v).toDouble(&b);
				if (!b) {
					cout << QObject::tr("%1 : this coordinate is not a number").arg(sommet.section(" ",6,6)).toStdString() << endl;
					return false;
				}
			}

			QString fichiernew = resultDir + QString("Orientation-%1.xml").arg(image);
			QFile newFile(fichiernew);
			if (!newFile.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) {
				cout << conv(QObject::tr("Fail to save file %1.")).arg(fichiernew).toStdString() << endl;
				return false;
			}
			QXmlStreamWriter xmlWriter(&newFile);
			xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne
			xmlWriter.writeTextElement(QString("Centre"), QString("%1 %2 %3").arg(x).arg(y).arg(z));
			if (doRot) {
				xmlWriter.writeStartElement(QString("ParamRotation"));
				xmlWriter.writeTextElement(QString("CodageAngulaire"), QString("%1 %2 %3").arg(roll).arg(pitch).arg(direction));
				xmlWriter.writeEndElement();	//ParamRotation
			}
			newFile.close();
		}
		oldFile.close();
		if (images.count()==0) {
			cout << conv(QObject::tr("No images found\n")).toStdString() << endl;
			return false;
		}
	}
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//fichier pointant vers les orientations initiales
FichierOriInit::FichierOriInit() {}

bool FichierOriInit::ecrire (const QString& fichier) {
	QFile file(fichier);
	if (!file.open(QFile::WriteOnly | QFile::Text | QFile::Truncate)) {
		cout << QObject::tr("Fail to open file %1.").arg(fichier).toStdString() << endl;
		return false;
	}
	QXmlStreamWriter xmlWriter(&file);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	xmlWriter.writeStartElement(QString("BDD_Orient"));
		xmlWriter.writeTextElement(QString("Id"), QString("Key-Ori-Initiale"));
		xmlWriter.writeTextElement(QString("KeySet"), QString("Key-Set-All-OrInitiale"));
		xmlWriter.writeTextElement(QString("KeyAssoc"), QString("Key-Assoc-Im2OrInitiale"));
	xmlWriter.writeEndElement();	//BDD_Orient


	file.close();
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


FichierCleCalib::FichierCleCalib() {}

bool FichierCleCalib::ecrire (const QString& fichier, const QList<int>& calib) {
	QFile oldFile(fichier);
	if (oldFile.exists()) {
		oldFile.remove();
	}

	QFile newFile(fichier);
	if (!newFile.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) {
		return false;
	}
	QXmlStreamWriter xmlWriter(&newFile);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne
	if (calib.count()>0) {
		QString s = (calib.count()==1)? QString() : QString("( ");
		for (int i=0; i<calib.count(); i++) {
			QString f = QVariant(calib.at(i)).toString();
			while (f.count()<3)
				f = QString("0")+f;
			s += QString("TheKeyCalib_%1 ").arg(f);
			if (i<calib.count()-1) s += QString("| ");
		}
		s += (calib.count()==1)? QString() : QString(")");
		xmlWriter.writeTextElement(QString("PatternNameApply"), s);
	} else
		xmlWriter.writeTextElement(QString("PatternNameApply"), QString("aucune"));
	xmlWriter.writeEndDocument();
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


FichierContraintes::FichierContraintes() {}

bool FichierContraintes::ecrire (const QString& fichier, bool classique, bool fisheye) {
	QFile oldFile(fichier);
	if (oldFile.exists()) {
		oldFile.remove();
	}

	QFile newFile(fichier);
	if (!newFile.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) {
		return false;
	}
	QXmlStreamWriter xmlWriter(&newFile);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

		xmlWriter.writeStartElement(QString("ContraintesCamerasInc"));
		if (classique) {
			xmlWriter.writeTextElement(QString("Val"), QString("eLiberteFocale_0"));
			xmlWriter.writeTextElement(QString("Val"), QString("eLib_PP_CD_00"));
			xmlWriter.writeTextElement(QString("Val"), QString("eLiberte_DR0"));
		}
		if (fisheye) {
			xmlWriter.writeTextElement(QString("Val"), QString("eLiberte_Dec0"));
			xmlWriter.writeTextElement(QString("Val"), QString("eLiberteParamDeg_0"));
		}
		xmlWriter.writeEndElement();	//ContraintesCamerasInc

	xmlWriter.writeEndDocument();
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//cas multi-échelle, étape 2 : écriture des poses figées (courtes focales)
FichierPosesFigees::FichierPosesFigees() {}

bool FichierPosesFigees::ecrire (const QString& fichier, const QList<int>& calibFigees, const QList<std::pair<QString, int> >& calibFiles, const QVector<ParamImage>& imgs, const QStringList& imgOri, const QString& maitresse, bool figees) {
	if (calibFiles.count()==0) return false;
	QFile oldFile(fichier);
	if (oldFile.exists()) {
		oldFile.remove();
	}
	QFile newFile(fichier);
	if (!newFile.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) {
		return false;
	}
	if (calibFigees.count()==0) return true;
	QXmlStreamWriter xmlWriter(&newFile);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	for (int i=0; i<imgs.count(); i++) {
		if (maitresse==imgs.at(i).getImageTif()) continue;	//pas l'image maîtresse
		if (!imgOri.contains(imgs.at(i).getImageTif())) continue;	//uniquement les images orientables
			//recherche de la focale correspondante
			QString calib = imgs.at(i).getCalibration();
			int j = 0;
			while (j<calibFiles.count()) {
				if (calib==calibFiles.at(j).first) break;
				j++;			
			}
		if (!calibFigees.contains(calibFiles.at(j).second)) continue;	//uniquement les courtes focales

		xmlWriter.writeStartElement(QString("ContraintesPoses"));
			xmlWriter.writeTextElement(QString("ByPattern"), QString("false"));
			xmlWriter.writeTextElement(QString("NamePose"), imgs.at(i).getImageTif());
			if (figees)
				xmlWriter.writeTextElement(QString("Val"), QString("ePoseFigee"));
			else
				xmlWriter.writeTextElement(QString("Val"), QString("ePoseLibre"));
		xmlWriter.writeEndElement();	//ContraintesPoses
	}

	xmlWriter.writeEndDocument();
	return true;
}

bool FichierPosesFigees::ecrire (const QString& fichier, const QVector<bool>& calibDissoc, const QList<std::pair<QString, int> >& calibFiles, const QVector<ParamImage>& imgs, const QStringList& imgOri, const QString& maitresse) {
	QList<int> calibFigees;
	for (int i=0; i<calibDissoc.count(); i++) {
		if (!calibDissoc.at(i))
			calibFigees.push_back(calibFiles.at(i).second);
	}
	bool b = ecrire (fichier, calibFigees, calibFiles, imgs, imgOri, maitresse, true);
	return b;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//liste des images utilisées pour créer les cartes de profondeur
FichierCartes::FichierCartes () {}

bool FichierCartes::ecrire(const QString& fichier, const CarteDeProfondeur& carte) {
	if (carte.getImagesCorrel().count()==0) return false;
	deleteFile(fichier);
	QFile file(fichier);
	if (!file.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) {
			return false;
	}
	QXmlStreamWriter xmlWriter(&file);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	xmlWriter.writeStartElement(QString("Images"));
		xmlWriter.writeTextElement(QString("Im1"), carte.getImageDeReference());
		for (int i=0; i<carte.getImagesCorrel().count(); i++) {
			if (carte.getImagesCorrel().at(i).isEmpty() || carte.getImagesCorrel().at(i)==carte.getImageDeReference()) continue;
			xmlWriter.writeTextElement(QString("ImPat"), carte.getImagesCorrel().at(i));
		}
	xmlWriter.writeEndElement();	//Images

	xmlWriter.writeEndDocument();
	return true;
}

QString FichierCartes::lire(const QString& fichier, CarteDeProfondeur& carte) {
	QString sbase = conv(QObject::tr("Depth map parameter file %1 reading :\n")).arg(fichier);
	XmlTree xmlTree(fichier);
	QString err = xmlTree.lire();
	if (!err.isEmpty()) return err;

	bool ok;
	XmlTag tag0;
	XmlTag& tag = *(&tag0);
	if (xmlTree.getMainTag().getNom()!=QString("Images")) return sbase+conv(QObject::tr("Unknown file format."));
	tag = xmlTree.getMainTag().getEnfant("Im1",&ok);
	if (!ok) return sbase+conv(QObject::tr("Reference image not found."));
	QString imageLue = tag.getContenu();
	if (carte.getImageDeReference()!=imageLue) return sbase+conv(QObject::tr("Reference image does not match to this depth map."));

	const QList<const XmlTag*> imgsListe = xmlTree.getMainTag().getEnfants("ImPat",&ok);
	if (!ok) return sbase+conv(QObject::tr("No images for correlation were found."));
	carte.modifImagesCorrel().clear();
	for (QList<const XmlTag*>::const_iterator it=imgsListe.begin(); it!=imgsListe.end(); it++) {
		carte.modifImagesCorrel().push_back((*it)->getContenu());
	}	
	return QString();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//fichier xml de référencement du masque saisi pour MicMac
FichierMasque::FichierMasque () {}

bool FichierMasque::ecrire(const QString& fichier, const ParamMasqueXml& paramMasqueXml) {
	QFile oldFile(fichier);
	if (oldFile.exists()) {
		oldFile.remove();
	}
	QFile masqueFile(fichier);
	if (!masqueFile.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) return false;
	QXmlStreamWriter xmlWriter(&masqueFile);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne
	//xmlWriter.writeCharacters(QString("<?xml version=\"1.0\" ?>"));

	xmlWriter.writeStartElement(QString("FileOriMnt"));
		xmlWriter.writeTextElement(QString("NameFileMnt"), paramMasqueXml.getNameFileMnt());
		if (!paramMasqueXml.getNameFileMasque().isEmpty()) xmlWriter.writeTextElement(QString("NameFileMasque"), paramMasqueXml.getNameFileMasque());
		xmlWriter.writeTextElement(QString("NombrePixels"), QVariant(paramMasqueXml.getNombrePixels().width()).toString()+QString(" ")+QVariant(paramMasqueXml.getNombrePixels().height()).toString());
		xmlWriter.writeTextElement(QString("OriginePlani"), QVariant(paramMasqueXml.getOriginePlani().x()).toString()+QString(" ")+QVariant(paramMasqueXml.getOriginePlani().y()).toString());
		xmlWriter.writeTextElement(QString("ResolutionPlani"), QVariant(paramMasqueXml.getResolutionPlani().x()).toString()+QString(" ")+QVariant(paramMasqueXml.getResolutionPlani().y()).toString());
		xmlWriter.writeTextElement(QString("OrigineAlti"), QVariant(paramMasqueXml.getOrigineAlti()).toString());
		xmlWriter.writeTextElement(QString("ResolutionAlti"), QVariant(paramMasqueXml.getResolutionAlti()).toString());
		xmlWriter.writeTextElement(QString("Geometrie"), paramMasqueXml.getGeometrie());
	xmlWriter.writeEndElement();	//FileOriMnt

	xmlWriter.writeEndDocument();
	masqueFile.close();

	//écriture de l'en-tête
/*	if (!masqueFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
		return false;
	}
	QTextStream inStream(&masqueFile);
	QFile newFile("tempo");
	if (!newFile.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
		return false;
	}
	QTextStream outStream(&newFile);
	QString text = QString("<?xml version=\"1.0\" ?>");
	text += inStream.readAll();
	outStream << text;
	masqueFile.close();
	newFile.close();
	masqueFile.remove();
	newFile.rename(fichier);	*/
	return true;
}

QString FichierMasque::lire(const QString& fichier, ParamMasqueXml& paramMasqueXml) {
	QString sbase = conv(QObject::tr("Mask referencing file %1 reading :\n")).arg(fichier);
	XmlTree xmlTree(fichier);
	QString err = xmlTree.lire();
	if (!err.isEmpty()) return err;

	bool ok;
	XmlTag tag0;
	XmlTag& tag = *(&tag0);
	if (xmlTree.getMainTag().getNom()!=QString("FileOriMnt")) return sbase+conv(QObject::tr("This file is not a referencing file."));

	tag = xmlTree.getMainTag().getEnfant("NameFileMnt",&ok);
	if (!ok) return sbase+conv(QObject::tr("DTM not found."));
	paramMasqueXml.setNameFileMnt( tag.getContenu() );

	tag = xmlTree.getMainTag().getEnfant("NameFileMasque",&ok);
	if (!ok) return sbase+conv(QObject::tr("Mask not found."));
	paramMasqueXml.setNameFileMasque( tag.getContenu() );

	tag = xmlTree.getMainTag().getEnfant("NombrePixels",&ok);
	if (!ok) return sbase+conv(QObject::tr("Size not found."));
	QString text = tag.getContenu();
	int w = text.section(" ",0,0).toInt(&ok);
	if (!ok) return sbase+conv(QObject::tr("Width is unvalid."));
	int h = text.section(" ",1,1).toInt(&ok);
	if (!ok) return sbase+conv(QObject::tr("Height is unvalid."));
	paramMasqueXml.setNombrePixels(QSize(w,h));

	tag = xmlTree.getMainTag().getEnfant("OriginePlani",&ok);
	if (!ok) return sbase+conv(QObject::tr("Planimetric origin not found."));
	text = tag.getContenu();
	double x0 = text.section(" ",0,0).toDouble(&ok);
	if (!ok) return sbase+conv(QObject::tr("X coordinate of planimetric origin is unvalid."));
	double y0 = text.section(" ",1,1).toDouble(&ok);
	if (!ok) return sbase+conv(QObject::tr("Y coordinate of planimetric origin is unvalid."));
	paramMasqueXml.setOriginePlani( QPointF(x0,y0) );

	tag = xmlTree.getMainTag().getEnfant("ResolutionPlani",&ok);
	if (!ok) return sbase+conv(QObject::tr("Planimetric resolution not found."));
	text = tag.getContenu();
	double dx = text.section(" ",0,0).toDouble(&ok);
	if (!ok) return sbase+conv(QObject::tr("X coordinate of planimetric resolution is unvalid."));
	double dy = text.section(" ",1,1).toDouble(&ok);
	if (!ok) return sbase+conv(QObject::tr("Y coordinate of planimetric resolution is unvalid."));
	paramMasqueXml.setResolutionPlani( QPointF(dx,dy) );

	tag = xmlTree.getMainTag().getEnfant("OrigineAlti",&ok);
	if (!ok) return sbase+conv(QObject::tr("Altimetric origin not found."));
	text = tag.getContenu();
	double z0 = text.section(" ",0,0).toDouble(&ok);
	if (!ok) return sbase+conv(QObject::tr("Altimetric origin is unvalid."));
	paramMasqueXml.setOrigineAlti( z0 );

	tag = xmlTree.getMainTag().getEnfant("ResolutionAlti",&ok);
	if (!ok) return sbase+conv(QObject::tr("Altimetric resolution not found."));
	text = tag.getContenu();
	double dz = text.section(" ",0,0).toDouble(&ok);
	if (!ok) return sbase+conv(QObject::tr("Altimetric resolution is unvalid."));
	paramMasqueXml.setResolutionAlti( dz );

	tag = xmlTree.getMainTag().getEnfant("Geometrie",&ok);
	if (!ok) return sbase+conv(QObject::tr("Mask not found."));
	paramMasqueXml.setGeometrie( tag.getContenu() );
			
	return QString();
}

///////////////////////////////////////////////////////////////////////////////////////////


//fichier xml de référencement du masque saisi pour MicMac
FichierDefMasque::FichierDefMasque () {}

bool FichierDefMasque::ecrire(QString dossier, QString fichier, QString masque, QString ref) {
	QFile file(dossier+fichier);
	if (!file.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) {
			return false;
	}
	QXmlStreamWriter xmlWriter(&file);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	xmlWriter.writeStartElement(QString("Planimetrie"));
		xmlWriter.writeStartElement(QString("MasqueTerrain"));
			xmlWriter.writeTextElement(QString("MT_Image"), masque);
			xmlWriter.writeTextElement(QString("MT_Xml"), ref);
		xmlWriter.writeEndElement();	//MasqueTerrain
	xmlWriter.writeEndElement();	//Planimetrie

	xmlWriter.writeEndDocument();
	return true;
}

QString FichierDefMasque::lire(QString dossier, QString fichier, QString & masque, QString & refmasque) {
/*	QString sbase = conv(QObject::tr("Lecture du fichier de définition du masque %1 :\n")).arg(fichier);
	XmlTree xmlTree(fichier);
	QString err = xmlTree.lire();
	if (!err.isEmpty()) return err;

	bool ok;
	XmlTag tag0;
	XmlTag& tag = *(&tag0);
	if (xmlTree.getMainTag().getNom()!=QString("Planimetrie")) return sbase+conv(QObject::tr("Ce n'est pas un fichier de définition."));
	const XmlTag* maskTag = &(xmlTree.getMainTag().getEnfant("MasqueTerrain",&ok));
	if (!ok) return sbase+conv(QObject::tr("Les paramètres n'ont pas été trouvés."));

	tag = maskTag->getEnfant("MT_Image",&ok);
	if (!ok) return sbase+conv(QObject::tr("Le masque n'a pas été trouvé."));
	masque = tag.getContenu().section("/",-1,-1);

	tag = maskTag->getEnfant("MT_Xml",&ok);
	if (!ok) return sbase+conv(QObject::tr("Le fichier de référencement du masque n'a pas été trouvé."));
	refmasque = tag.getContenu().section("/",-1,-1);

	QString image;
	QString lecture = FichierMasque::lire(dossier+refmasque,image);	//image : sans le dossier
	if (!lecture.isEmpty()) return lecture;

	if (image!=masque)	//il faut que les noms des masques soient cohérents
		return sbase+conv(QObject::tr("Le masque indiqué dans le fichier de définition du masque n'est pas le même que celui indiqué dans le fichier de référencement."));
		*/
	return QString();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//repère de corrélation
FichierRepere::FichierRepere () {}

bool FichierRepere::ecrire(const QString& fichier, double profondeur) {
	QFile file(fichier);
	if (!file.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) return false;
	QXmlStreamWriter xmlWriter(&file);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	xmlWriter.writeStartElement(QString("RepereLoc"));
		xmlWriter.writeStartElement(QString("RepereCartesien"));
			xmlWriter.writeTextElement(QString("Ori"), QString("0 0 %1").arg(profondeur));
			xmlWriter.writeTextElement(QString("Ox"), QString("1 0 0"));
			xmlWriter.writeTextElement(QString("Oy"), QString("0 1 0"));
			xmlWriter.writeTextElement(QString("Oz"), QString("0 0 1"));
		xmlWriter.writeEndElement();	//RepereCartesien
	xmlWriter.writeEndElement();	//RepereLoc

	xmlWriter.writeEndDocument();
	return true;
}

QString FichierRepere::lire(const QString& fichier) {
	QString sbase = conv(QObject::tr("Frame file %1 reading :\n")).arg(fichier);
	XmlTree xmlTree(fichier);
	QString err = xmlTree.lire();
	if (!err.isEmpty()) return err;

	bool ok;
	XmlTag tag0;
	XmlTag& tag = *(&tag0);
	if (xmlTree.getMainTag().getNom()!=QString("RepereLoc")) return sbase+conv(QObject::tr("This file is not a frame file."));
	const XmlTag* repTag = &(xmlTree.getMainTag().getEnfant("RepereCartesien",&ok));
	if (!ok) return sbase+conv(QObject::tr("Parameters not found."));

	QStringList l, m1, m2;
	l << "Ori" << "Ox" << "Oy" << "Oz";
	m1 << conv(QObject::tr("Frame origin not found."))
	   << conv(QObject::tr("X axis not found."))
	   << conv(QObject::tr("Y axis not found."))
	   << conv(QObject::tr("Z axis not found."));
	m2 << conv(QObject::tr("A coordinate of the origin is unvalid."))
	   << conv(QObject::tr("A coordinate of the x axis is unvalid."))
	   << conv(QObject::tr("A coordinate of the y axis is unvalid."))
	   << conv(QObject::tr("A coordinate of the z axis is unvalid."));
	for (int k=0; k<l.count(); k++) {
		tag = repTag->getEnfant(l.at(k),&ok);
		if (!ok) return sbase+m1.at(k);
		QString text = tag.getContenu();
		for (int i=0; i<3; i++) {
			/*float c = */text.section(" ",i,i).toDouble(&ok);
			if (!ok) return sbase+m2.at(k);
		}
	}
	return QString();
}

///////////////////////////////////////////////////////////////////////////////////////////


//intervalle de recherche pour la corrélation
FichierIntervalle::FichierIntervalle() {}

bool FichierIntervalle::ecrire(const QString& fichier, float pmin, float pmax) {
	QFile file(fichier);
	if (!file.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) {
		return false;
	}
	QXmlStreamWriter xmlWriter(&file);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	xmlWriter.writeStartElement(QString("IntervSpecialZInv"));
	xmlWriter.writeTextElement(QString("MulZMin"), QVariant(pmin).toString());
	xmlWriter.writeTextElement(QString("MulZMax"), QVariant(pmax).toString());
	xmlWriter.writeEndElement();	//IntervSpecialZInv

	xmlWriter.writeEndDocument();
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////


//filtrage des discontinuités et fortes pentes
FichierDiscontinuites::FichierDiscontinuites() {}

bool FichierDiscontinuites::ecrire(const QString& fichier, float seuil, float coeff) {
	QFile file(fichier);
	if (!file.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) {
		return false;
	}
	QXmlStreamWriter xmlWriter(&file);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	xmlWriter.writeTextElement(QString("SeuilAttenZRegul"), QVariant(seuil).toString());
	xmlWriter.writeTextElement(QString("AttenRelatifSeuilZ"), QVariant(coeff).toString());

	xmlWriter.writeEndDocument();
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////


//définit le nom des cartes de profondeur
FichierNomCarte::FichierNomCarte() {}

bool FichierNomCarte::ecrire(const QString& fichier, const QString& numCarte, bool TA, bool repereImg) {
	QFile file(fichier);
	if (!file.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) {
		return false;
	}
	QXmlStreamWriter xmlWriter(&file);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	QString s = (TA)? QString("TA%1/").arg(numCarte) : (!repereImg)? QString("GeoTer%1/").arg(numCarte): QString("GeoI%1/").arg(numCarte);
	xmlWriter.writeTextElement(QString("TmpMEC"), s);
	xmlWriter.writeTextElement(QString("TmpPyr"), s);
	xmlWriter.writeTextElement(QString("TmpResult"), s);
	QString s2 = QString("Geom-Im-%1").arg(numCarte);
	xmlWriter.writeTextElement(QString("NomChantier"), s2);

	xmlWriter.writeEndDocument();
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////


//paramètres des orthoimages
FichierOrtho::FichierOrtho() {}

bool FichierOrtho::ecrire(const QString& fichier, const CarteDeProfondeur& carte, const ParamMain& paramMain) {
	QFile file(fichier);
	if (!file.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) {
		return false;
	}
	QXmlStreamWriter xmlWriter(&file);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	xmlWriter.writeStartElement(QString("ImageSelecteur"));
		xmlWriter.writeTextElement(QString("ModeExclusion"), QString("false"));
			QString s("(");
			s += carte.getImgsOrtho().first();
			for (QStringList::const_iterator it=carte.getImgsOrtho().begin()+1; it!=carte.getImgsOrtho().end(); it++)
				s += QString ("|") + *it;
			s += QString(")");
		xmlWriter.writeTextElement(QString("PatternSel"), s);
	xmlWriter.writeEndElement();	//ImageSelecteur

	QString num = paramMain.getNumImage(carte.getImageDeReference());

	xmlWriter.writeStartElement(QString("GenerePartiesCachees"));
		xmlWriter.writeTextElement(QString("ByMkF"), QString("true"));
		xmlWriter.writeTextElement(QString("SeuilUsePC"), QString("3"));
		xmlWriter.writeTextElement(QString("KeyCalcPC"), QString("NKS-Assoc-AddDirAndPref@ORTHO%1@PC_").arg(num));
		xmlWriter.writeTextElement(QString("PatternApply"), s);
		xmlWriter.writeStartElement(QString("MakeOrthoParImage"));
			xmlWriter.writeTextElement(QString("OrthoBiCub"), QString("-1.5"));
			xmlWriter.writeTextElement(QString("ResolRelOrhto"), QVariant(carte.getEchelleOrtho()).toString());
			xmlWriter.writeTextElement(QString("KeyCalcInput"), QString("Key-Assoc-Id_couleur"));
			xmlWriter.writeTextElement(QString("KeyCalcOutput"), QString("NKS-Assoc-AddDirAndPref@ORTHO%1@Ort_").arg(num));
			xmlWriter.writeTextElement(QString("KeyCalcIncidHor"), QString("NKS-Assoc-AddDirAndPref@ORTHO%1@Incid_").arg(num));
			xmlWriter.writeTextElement(QString("ResolIm"), QString("1.0"));
			xmlWriter.writeTextElement(QString("CalcIncAZMoy"), QString("true"));
		xmlWriter.writeEndElement();	//MakeOrthoParImage
	xmlWriter.writeEndElement();	//GenerePartiesCachees

	xmlWriter.writeEndDocument();
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////


//définit le nom des orthoimages simples en entrée du mosaïcage
FichierPorto::FichierPorto() {}

bool FichierPorto::ecrire(const QString& fichier, const QString& num) {
	QFile file(fichier);
	if (!file.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate)) {
		return false;
	}
	QXmlStreamWriter xmlWriter(&file);
	xmlWriter.setAutoFormatting(true);	//espaces et retour à la ligne

	xmlWriter.writeTextElement(QString("FileMNT"), QString("../GeoTer%1/Z_Num7_DeZoom1_Geom-Im-%1.xml").arg(num));

	xmlWriter.writeEndDocument();
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////


//fichier de géoréférencement des cartes, MNT, TA et orthoimages
FichierGeorefMNT::FichierGeorefMNT() {}

QString FichierGeorefMNT::lire(GeorefMNT& georefMNT) {
	QString sbase = conv(QObject::tr("DTM georeferencing file %1 reading :\n")).arg(georefMNT.getFichier());
	XmlTree xmlTree(georefMNT.getFichier());
	QString err = xmlTree.lire();
	if (!err.isEmpty()) return err;

	bool ok;
	XmlTag tag0;
	XmlTag& tag = *(&tag0);
	if (xmlTree.getMainTag().getNom()!=QString("FileOriMnt")) return sbase+conv(QObject::tr("DTM georeferencing file %1 reading :\n"));

	tag = xmlTree.getMainTag().getEnfant("OriginePlani",&ok);
	if (!ok) return sbase+conv(QObject::tr("Planimetric origin not found."));
	QString text = tag.getContenu();
	double x0 = text.section(" ",0,0).toDouble(&ok);
	if (!ok) return sbase+conv(QObject::tr(" planimetric origin (x0) is unvalid."));
	georefMNT.setX0(x0);
	double y0 = text.section(" ",1,1).toDouble(&ok);
	if (!ok) return sbase+conv(QObject::tr(" planimetric origin (y0) is unvalid."));
	georefMNT.setY0(y0);

	tag = xmlTree.getMainTag().getEnfant("ResolutionPlani",&ok);
	if (!ok) return sbase+conv(QObject::tr("Planimetric resolution not found."));
	text = tag.getContenu();
	double dx = text.section(" ",0,0).toDouble(&ok);
	if (!ok) return sbase+conv(QObject::tr(" planimetric resolution (dx) is unvalid."));
	georefMNT.setDx(dx);
	double dy = text.section(" ",1,1).toDouble(&ok);
	if (!ok) return sbase+conv(QObject::tr(" planimetric resolution (dy) is unvalid."));
	georefMNT.setDy(dy);

	return QString();
}

///////////////////////////////////////////////////////////////////////////////////////////



QString focaleRaw(const QString& image, const ParamMain& paramMain, MetaDataRaw& metaDataRaw) {
	QString commande = QString("%1bin/ElDcraw -i -v %2 >%3truc").arg(noBlank(paramMain.getMicmacDir())).arg(noBlank(image)).arg(noBlank(paramMain.getDossier()));
	if (execute(commande)!=0)
		return (QObject::tr("Fail to extract image %1 metadata.")).arg(image);
	QFile file(paramMain.getDossier()+QString("truc"));
	if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
		return (QObject::tr("Fail to extract image %1 metadata.")).arg(image);
	QTextStream inStream(&file);
	while (!inStream.atEnd()) {
		QString text = inStream.readLine();
		if (text.contains("Camera:"))	//Camera: Canon EOS 350D DIGITAL
			metaDataRaw.camera = text.trimmed().simplified().section(" ",1,-1);
		else if (text.contains("Focal length:")) {	//Focal length: 55.0 mm
			QString str = text.trimmed().simplified().section(" ",2,2);
			bool ok = false;
			metaDataRaw.focale = str.toDouble(&ok);
			if (!ok)  {
				metaDataRaw.focale = 0;
				continue;
			}
		} else if (text.contains("Focal Equi35:")) {	//Focal Equi35: -1.0 mm
			QString str = text.trimmed().simplified().section(" ",2,2);
			bool ok = false;
			metaDataRaw.feq35 = str.toDouble(&ok);
			if (!ok)  {
				metaDataRaw.feq35 = 0;
				continue;
			}
		} else if (text.contains("Image size:")) {	//Image size:  3474 x 2314
			QStringList str = text.trimmed().simplified().split(" ",QString::SkipEmptyParts);
			bool ok = false;
			metaDataRaw.imgSize.setWidth(str[2].toDouble(&ok));
			if (!ok)  {
				metaDataRaw.imgSize.setWidth(0);
				continue;
			}
			metaDataRaw.imgSize.setHeight(str[4].toDouble(&ok));
			if (!ok)  {
				metaDataRaw.imgSize.setHeight(0);
				continue;
			}
		} else continue;
	}
	file.close();
	file.remove();
	return QString();
}

bool focaleTif(const QString& image, const QString& micmacDir, int* focale, QSize* taille)
{
	if (focale==0 && taille==0) return true;

	QString commande = QString("%1bin/tiff_info %2 > %3/tempofile.txt").arg(noBlank(micmacDir)).arg(noBlank(image)).arg(noBlank(image.section("/",0,-2)));
	if (execute(commande)!=0) {
		cout << QObject::tr("tiff_info binary failed to read image %1 metadata.").arg(image).toStdString() << endl;
		return false;
	}

	QFile file( image.section("/",0,-2)+QString("/tempofile.txt") );
	if (!file.open(QFile::ReadOnly | QFile::Text)) {
		cout << QObject::tr("Fail to open image %1 metadata file.").arg(image).toStdString() << endl;
		file.remove();
		return false;
	}
	QTextStream instream(&file);

	while (!instream.atEnd()) {
		QString text = instream.readLine().trimmed().simplified();
		if ((focale==0 || !text.contains("FocMm")) && (taille==0 || !text.contains("SIZE"))) continue;
		bool ok;
		if (text.contains("FocMm")) {
			QString foc = text.section(" ",1,1);
			if (!foc.at(foc.count()-1).isDigit()) foc = foc.left(foc.count()-1);
			int f = foc.toInt(&ok);

			// focal value is not defined in the file's meta-data
			// try to retrieve the value from image filename
			if ( !ok || f==-1 ){
				ok = false;
				string directory, filename, focalStr;
				SplitDirAndFile( directory, filename, image.toStdString() );
				size_t underscore_pos = filename.find( '_' );
				if ( underscore_pos==string::npos || filename.size()<2 ) break;
				f = atoi( filename.substr( 1, underscore_pos-1 ).c_str() );
				if ( f!=0 ) ok=true; 
			}

			if ( !ok ){
				cout << QObject::tr("The focal length extracted from image %1 metadata is uncorrect.").arg(image).toStdString() << endl;
				file.remove();
				return false;
			}
			*focale = f;
		} else {
			QString s = text.section(" ",1);
			s = s.section(" ",1,1);
			QString x = s.section(",",0,0);
			QString y = s.section(",",1,1);
			
			if (x.left(1)==QString("(")) x = x.right(x.count()-1);
			if (y.right(2)==QString(");")) y = y.left(y.count()-2);
			int w = x.toInt(&ok);
			if (!ok) {
				cout << QObject::tr("The image width extracted from image %1 metadata is uncorrect.").arg(image).toStdString() << endl;
				file.remove();
				return false;
			}
			int h = y.toInt(&ok);
			if (!ok) {
				cout << QObject::tr("The image height extracted from image %1 metadata is uncorrect.").arg(image).toStdString() << endl;
				file.remove();
				return false;
			}

			taille->setWidth(max(w,h));
			taille->setHeight(min(w,h));
		}
	}

	file.remove();
	return true;
}

// get meta data of an image with an extracting tool
void getMetaData( const string &i_directory, const string &i_image, const string &i_tool, int &o_focal, QSize &o_size )
{
	// get a valid name for the tool
	ExternalToolItem toolItem =  g_externalToolHandler.get( i_tool );
	if ( toolItem.isCallable() ) return;
	
	// run the tool in with a system call
	string imageFullName = i_directory+'/'+i_image,
		   tempFileName  = i_directory+"/truc";
	QString commande( ( toolItem.callName() + ' ' + imageFullName + " > " + tempFileName ).c_str() );
	if ( execute(commande)!=EXIT_SUCCESS )	return;

	// open temporary file containing meta data
	QFile file( QString( tempFileName.c_str() ) );
	if (!file.open(QIODevice::ReadOnly | QIODevice::Text)){
		#ifdef _DEBUG
			cerr << "getMetaData (" << i_directory << ',' << i_image << ',' << i_tool << ") failed to open " << tempFileName << endl;
		#endif
		return;
	}
	
	int colonPos, mmPos, xPos;
	QString sectionName, sectionValue;
	QTextStream inStream( &file );
	while ( !inStream.atEnd() )
	{
		QString text = inStream.readLine();

		colonPos = text.indexOf( ":" );
		if ( colonPos==-1 ) continue; // this is not a valid section
		sectionName = text.left( colonPos ).trimmed().toLower();
		
		if ( ( sectionName=="focal length") && ( o_focal==-1 ) )
		{
			mmPos = text.indexOf( "mm" );
			if ( mmPos==-1 ) continue; // this is not a valid section

			// extract length value from section's value
			o_focal = atoi( text.mid( colonPos+1, mmPos-colonPos-1 ).toStdString().c_str() );

			if ( o_focal!=-1 && o_size.width()!=-1 ) return;
		}
		else if ( ( sectionName=="image size" ) && ( o_size.width()==-1 ) )
		{
			sectionValue = text.right( text.size()-colonPos-2 );
			// at this point sectionValue should be of the form "WIDHTxHEIGHT" or "WIDTH x HEIGHT"
			xPos = sectionValue.indexOf( "x" );
			
			o_size.setWidth( atoi( sectionValue.left( xPos ).toStdString().c_str() ) );
			o_size.setHeight( atoi( sectionValue.right( sectionValue.size()-xPos-1 ).toStdString().c_str() ) );

			if ( o_focal!=-1 && o_size.width()!=-1 ) return;
		}
	}
}

// get focal and image size for EVIF/XMP data
// extract meta-data with external tools exif and exiv2
bool focaleOther( const string &i_directory, const string &i_image, int &o_focal, QSize &o_size )
{
	// value defined as invalid
	o_focal = -1;
	o_size.setWidth( -1 );

	getMetaData( i_directory, i_image, "exiv2", o_focal, o_size );

	if ( ( o_focal!=-1 ) && ( o_size.width()!=-1 ) ) return true;

	#ifdef _DEBUG
		cerr << "focaleOther (" << i_directory << ',' << i_image << ") failed to recover meta data with exiv2" << endl;
	#endif

	getMetaData( i_directory, i_image, "exiftool", o_focal, o_size );
	
	#ifdef _DEBUG
		if ( ( o_focal==-1 ) || ( o_size.width()==-1 ) )
			cerr << "focaleOther (" << i_directory << ',' << i_image << ") failed to recover meta data with exiftool" << endl;
	#endif

	return ( ( o_focal!=-1 ) && ( o_size.width()!=-1 ) );
}

bool createEmptyFile(const QString& fichier) {
	if (QFile(fichier).exists()) QFile(fichier).remove();
	QFile file(fichier);
	if (!file.open(QFile::WriteOnly | QFile::Text | QIODevice::Truncate))
		return false;
	file.close();
	return true;
}
