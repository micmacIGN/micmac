#include "MatrixManager.h"

HistoryManager::HistoryManager():
    _actionIdx(0),
    _initSize(0)
{}

void HistoryManager::push_back(selectInfos &infos)
{
    int sz = _infos.size();

    if (_actionIdx < sz)
    {
        for (int aK=_actionIdx; aK < sz; ++aK)
            _infos.pop_back();
    }

    _infos.push_back(infos);

    _actionIdx++;
}

bool HistoryManager::load(QString filename)
{
    _infos.clear();
    _filename  = filename;

    QFile file(_filename);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        return false;
    }

    QDomDocument doc;
    if (!doc.setContent(&file))
    {
        file.close();
        return false;
    }
    file.close();

    QDomNode n = doc.firstChild(); //Node SelectionInfo

    QDomNodeList nl = n.childNodes();

    for (int aK=0; aK <nl.size();++aK)
    {
        n = nl.at(aK); //Node Item

        QDomNodeList nl2 = n.childNodes();

        selectInfos SInfo;
        QVector<QPointF> pts;

        for (int bK=0; bK <nl2.size();++bK)
        {
            QDomNode n2 = nl2.at(bK);

            QDomElement e = n2.toElement();
            QString name = e.tagName();
            QDomText t = e.firstChild().toText();

            if (!t.isNull())
            {
                QString data = t.data();
                QStringList strList = data.split(" ");

                if(name == "ModelViewMatrix")
                {
                    for (int cK=0; cK<strList.size();++cK)
                        SInfo.mvmatrix[cK] = strList[cK].toFloat();
                }
                else if(name == "ProjMatrix")
                {
                    for (int cK=0; cK<strList.size();++cK)
                        SInfo.projmatrix[cK] = strList[cK].toFloat();
                }
                else if(name == "glViewport")
                {
                    for (int cK=0; cK<strList.size();++cK)
                        SInfo.glViewport[cK] = strList[cK].toFloat();
                }
                else if(name == "Pt")
                {
                    if (strList.size() == 2)
                       pts.push_back(QPointF(strList[0].toFloat(),strList[1].toFloat()));
                }
                else if(name == "Mode")
                    SInfo.selection_mode = data.toInt();
            }
        }
        SInfo.poly = pts;
        _infos.push_back(SInfo);
    }
    _initSize  = _infos.size();
    _actionIdx = _infos.size();
    return true;
}

void HistoryManager::save()
{
    //std::cout << "saving in " << _filename.toStdString().c_str() << std::endl;

    QFile outFile(_filename);
    if (!outFile.open(QIODevice::WriteOnly)) return;

    QString tmp = _filename;
    QString pts3d_file = _filename.replace("selectionInfo", "polyg3d");
    _filename = tmp;

    QFile ptsFile(pts3d_file);
    if (!ptsFile.open(QIODevice::WriteOnly)) return;

    QDomDocument doc, doc2;
    QDomElement SI  = doc.createElement("SelectionInfo");
    QDomElement SI2 = doc2.createElement("Polyg3D");

    QDomText t;
    for (int i = 0; i < _infos.size(); ++i)
    {
        QDomElement SII            = doc.createElement("Item");
        QDomElement SIt            = doc2.createElement("Item");
        QDomElement mvMatrixElem   = doc.createElement("ModelViewMatrix");
        QDomElement ProjMatrixElem = doc.createElement("ProjMatrix");
        QDomElement glViewportElem = doc.createElement("glViewport");
        QDomElement Mode           = doc.createElement("Mode");
        QDomElement Mode2          = doc2.createElement("Mode");

        const selectInfos &SInfo = _infos[i];

        if ((SInfo.mvmatrix != NULL) && (SInfo.projmatrix != NULL) && (SInfo.glViewport != NULL))
        {
            QString text1, text2;

            text1 = QString::number(SInfo.mvmatrix[0], 'f');
            text2 = QString::number(SInfo.projmatrix[0], 'f');

            for (int aK=1; aK < 16;++aK)
            {
                text1 += " " + QString::number(SInfo.mvmatrix[aK], 'f');
                text2 += " " + QString::number(SInfo.projmatrix[aK], 'f');
            }

            t = doc.createTextNode(text1);
            mvMatrixElem.appendChild(t);

            t = doc.createTextNode(text2);
            ProjMatrixElem.appendChild(t);

            text1 = QString::number(SInfo.glViewport[0]) ;
            for (int aK=1; aK < 4;++aK)
                text1 += " " + QString::number(SInfo.glViewport[aK]);

            t = doc.createTextNode(text1);
            glViewportElem.appendChild(t);

            SII.appendChild(mvMatrixElem);
            SII.appendChild(ProjMatrixElem);
            SII.appendChild(glViewportElem);

            QVector <QPointF> pts = SInfo.poly;

            for (int aK=0; aK < pts.size(); ++aK)
            {
                QDomElement Point    = doc.createElement("Pt");
                QDomElement Pt3D     = doc.createElement("Pt");
                QString str = QString::number(pts[aK].x(), 'f',0) + " " + QString::number(pts[aK].y(), 'f',0);

                t = doc.createTextNode( str );
                Point.appendChild(t);
                SII.appendChild(Point);

                //Export points 3D
				QVector3D pt3d;
                MatrixManager *MM = new MatrixManager();
                MM->importMatrices(SInfo);
                MM->getInverseProjection(pt3d, pts[aK], 0.f);
				// str = QString::number(pt3d.x(), 'f') + " " + QString::number(pt3d.y(), 'f') + " " + QString::number(pt3d.z(), 'f');
//  MPD : tentative de ne pas perdre betement de la precision qui peut etre genant avec les grande coordonnees ? 
	        str = QString::number(pt3d.x(), 'f',20) + " " + QString::number(pt3d.y(), 'f',20) + " " + QString::number(pt3d.z(), 'f',20);
                t = doc2.createTextNode( str );
                Pt3D.appendChild(t);
                SIt.appendChild(Pt3D);
            }

            t = doc.createTextNode(QString::number(SInfo.selection_mode));
            Mode.appendChild(t);
            SII.appendChild(Mode);

            t = doc2.createTextNode(QString::number(SInfo.selection_mode));
            Mode2.appendChild(t);
            SIt.appendChild(Mode2);

            SI.appendChild(SII);
            SI2.appendChild(SIt);
        }
        else
            std::cerr << "saveSelectInfos: null matrix";
    }

    doc.appendChild(SI);
    doc2.appendChild(SI2);

    QTextStream content(&outFile);
    content.flush();
    content << doc.toString();
    outFile.close();

    QTextStream content2(&ptsFile);
    content2.flush();
    content2 << doc2.toString();
    ptsFile.close();
}
