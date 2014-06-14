#include "Tree.h"

#define HORSIMAGE "Outside image"

#define COLOR_OVER "#c89354"
#define NON_SAISIE "#ba5606"

ModelPointGlobal::ModelPointGlobal(QObject *parent, cAppli_SaisiePts *appli):
QAbstractTableModel(parent),
mAppli(appli),
_interface((cQT_Interface*)appli->Interface())
{
}

int ModelPointGlobal::rowCount(const QModelIndex & /*parent*/) const
{
    return AllCount();
}

int ModelPointGlobal::columnCount(const QModelIndex & /*parent*/) const
{
    return 2;
}

QVariant ModelPointGlobal::data(const QModelIndex &index, int role) const
{
    if (role == Qt::DisplayRole || role == Qt::EditRole)
    {
        if(index.row() < PG_Count())
        {
            cSP_PointGlob * pg = mAppli->PGlob(index.row());
            switch (index.column())
            {
            case 0:
                return QString("%1").arg(pg->PG()->Name().c_str());
            case 1:
            {
                Pt3dr *p3d = pg->PG()->P3D().PtrVal();
                return QString("%1\t %2\t %3")
                        .arg(QString::number(p3d->x, 'f' ,2))
                        .arg(QString::number(p3d->y, 'f' ,2))
                        .arg(QString::number(p3d->z, 'f' ,2));
            }
            }
        }
        else if (index.row() < AllCount())
        {
            int id = index.row() - PG_Count();
            if(id >= 0 && id < CaseNamePointCount())
            {
                cCaseNamePoint cnPt = mAppli->Interface()->GetCaseNamePoint(id);
                switch (index.column())
                {
                case 0:
                    return QString("%1").arg(cnPt.mName.c_str());
                case 1:
                    return QString(tr("Not measured"));
                }
            }
        }
    }

    if (role == Qt::BackgroundColorRole)
    {
        QColor selectPGlob  = QColor(COLOR_OVER);
        if(mAppli->PGlob(index.row()) == _interface->currentPGlobal() && _interface->currentPGlobal() && index.column() == 0)
                return selectPGlob;

        cSP_PointGlob * pg = mAppli->PGlob(index.row());

        if (pg != NULL && pg->getPointes().size())
        {
            QColor NonSaisie(NON_SAISIE);

            std::map<std::string,cSP_PointeImage *> ptIs = pg->getPointes();

            for
                    (
                     std::map<std::string,cSP_PointeImage *>::iterator itM = ptIs.begin();
                     itM!= ptIs.end();
                     itM++
                     )
            {
                cSP_PointeImage * ptImag = itM->second;
                if(ptImag->Saisie()->Etat() == eEPI_NonSaisi && ptImag->Visible())
                    return NonSaisie;
            }
        }
    }

    if (role == Qt::TextColorRole)
        if(mAppli->PGlob(index.row()) == _interface->currentPGlobal() && _interface->currentPGlobal())
                return QColor(Qt::white);

    return QVariant();
}

QVariant ModelPointGlobal::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (role == Qt::DisplayRole)
    {
        if (orientation == Qt::Horizontal)
        {
            switch (section)
            {
            case 0:
                return QString(tr("Point"));
            case 1:
                return QString(tr("Coordinates"));
            }
        }
    }

    return QVariant();
}

bool ModelPointGlobal::setData(const QModelIndex &index, const QVariant &value, int role)
{
    QString qnewName  = value.toString();

    if(qnewName == QString(""))
        return false;

    if (role == Qt::EditRole)
    {
        string oldName = mAppli->PGlob(index.row())->PG()->Name();
        string newName = qnewName.toStdString();

        mAppli->ChangeName(oldName, newName);

        emit pGChanged();
    }

    return true;
}

Qt::ItemFlags ModelPointGlobal::flags(const QModelIndex &index) const
{

    switch (index.column())
    {
    case 0:
        if(index.row() < PG_Count())
            return QAbstractTableModel::flags(index) | Qt::ItemIsEditable;
    case 1:
        return QAbstractTableModel::flags(index);
    }

    return QAbstractTableModel::flags(index);
}

bool ModelPointGlobal::insertRows(int row, int count, const QModelIndex &parent)
{
    beginInsertRows(QModelIndex(), row, row+count-1);
    endInsertRows();
    return true;
}

int ModelPointGlobal::AllCount() const
{
    return  PG_Count() + CaseNamePointCount();
}

int ModelPointGlobal::PG_Count() const
{
    return  mAppli->PG().size();
}

int ModelPointGlobal::CaseNamePointCount() const
{
    return  mAppli->Interface()->GetNumCaseNamePoint();
}

cAppli_SaisiePts *ModelPointGlobal::getMAppli() const
{
    return mAppli;
}

bool ModelPointGlobal::caseIsSaisie(int idRow)
{
    int idCase = idRow - PG_Count();

    QString nameCase(mAppli->Interface()->GetCaseNamePoint(idCase).mName.c_str());

    for (int i = 0; i < PG_Count(); ++i)
    {
       if(nameCase == QString(mAppli->PGlob(i)->PG()->Name().c_str()))
           return true;
    }

    return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ModelCImage::ModelCImage(QObject *parent, cAppli_SaisiePts *appli)
    :QAbstractTableModel(parent),
      mAppli(appli),
      _interface((cQT_Interface*)appli->Interface())
{

}

int ModelCImage::rowCount(const QModelIndex & /*parent*/) const
{
    return mAppli->imagesVis().size();
}

int ModelCImage::columnCount(const QModelIndex & /*parent*/) const
{
    return 3;
}

QVariant ModelCImage::data(const QModelIndex &index, int role) const
{

    if (role == Qt::DisplayRole || role == Qt::EditRole)
    {
        if(index.row() < (int) mAppli->imagesVis().size())
        {
            cImage* iImage = mAppli->imageVis(index.row());

            switch (index.column())
            {
            case 0:
                return QString("%1").arg(iImage->Name().c_str());
            case 1:
            {

                cSP_PointGlob* pg = _interface->currentPGlobal();

                if(!pg)
                    return QString("");

                cSP_PointeImage* pI = iImage->PointeOfNameGlobSVP(pg->PG()->Name());

                if(pI)
                {
                    cOneSaisie* cOS = pI->Saisie();
                    if(cOS)
                    {
                        eEtatPointeImage state = cOS->Etat();

                        switch (state)
                        {
                        case eEPI_NonSaisi:
                        {
                            if(pI->Visible())
                                return QString(tr("to validate"));
                            else
                                return QString(tr(HORSIMAGE));
                        }
                        case eEPI_Refute:
                            return QString(tr("refuted"));
                        case eEPI_Douteux:
                            return QString(tr("dubious"));
                        case eEPI_Valide:
                            return QString(tr("valid"));
                        case eEPI_NonValue:
                            return QString(tr("no value"));
                        case eEPI_Disparu:
                            return QString(tr("vanished"));
                        case eEPI_Highlight:
                            return QString(tr("highlighted"));
                        }
                    }
                }

                return QString(tr(HORSIMAGE));
            }
            case 2:
            {
                cSP_PointGlob* pg = _interface->currentPGlobal();

                if(!pg)
                    return QString("");

                cSP_PointeImage* pI = iImage->PointeOfNameGlobSVP(pg->PG()->Name());

                if (pI)
                {
                    cOneSaisie* cOS = pI->Saisie();

                    if(cOS)
                    {
                        if(cOS->Etat() == eEPI_Disparu)
                            return QString("");

                        return QString("%1\t %2")
                                .arg(QString::number(cOS->PtIm().x, 'f' ,1))
                                .arg(QString::number(cOS->PtIm().y, 'f' ,1));
                    }
                }

                return QString("");
            }
            }
        }
    }
    if (role == Qt::BackgroundColorRole)
    {

        QColor Red          = QColor("#87384c");
        QColor NonSaisie    = QColor(NON_SAISIE);
        QColor Douteux      = QColor("#a95b3b");
        QColor Valide       = QColor("#3c7355");
        QColor imageVisible = QColor("#3a819c");
        QColor selectPGlob  = QColor(COLOR_OVER);

        cSP_PointGlob* pg   = _interface->currentPGlobal();

        if(!pg)
            return QVariant(QColor("#5f5f5f"));

        cImage* iImage = mAppli->imageVis(index.row());

        if(index.column() == 0)
        {
            if(iImage == _interface->currentCImage() )
                return selectPGlob;
            else if (_interface->isDisplayed(iImage))
                return imageVisible;
        }

        cSP_PointeImage* pI = iImage->PointeOfNameGlobSVP(pg->PG()->Name());

        if(pI)
        {

            cOneSaisie* cOS = pI->Saisie();
            if(cOS)
            {
                eEtatPointeImage state = cOS->Etat();

                switch (state)
                {
                case eEPI_NonSaisi:
                {
                    if(pI->Visible())
                        return NonSaisie;
                    else
                        return Red;
                }
                case eEPI_Refute:
                    return Red;
                case eEPI_Douteux:
                    return Douteux;
                case eEPI_Valide:
                    return Valide;
                case eEPI_NonValue:
                    return Red;
                case eEPI_Disparu:
                    return Red;
                case eEPI_Highlight:
                    return Red;
                }
            }
        }

        return Red;

    }

    if (role == Qt::TextColorRole && index.column() == 0 && _interface->currentPGlobal())
            if(index.row() < (int)mAppli->imagesVis().size() && mAppli->imageVis(index.row()) == _interface->currentCImage() )
                return QColor(Qt::white);

    return QVariant();
}

QVariant ModelCImage::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (role == Qt::DisplayRole)
    {
        if (orientation == Qt::Horizontal) {
            switch (section)
            {
            case 0:
                return QString(tr("Image"));
            case 1:
                return QString(tr("State"));
            case 2:
                return QString(tr("Coordinates"));
            }
        }
    }
    return QVariant();
}

bool ModelCImage::setData(const QModelIndex &index, const QVariant &value, int role)
{
    return false;
}

Qt::ItemFlags ModelCImage::flags(const QModelIndex &index) const
{

    switch (index.column())
    {
    case 0:
        return QAbstractTableModel::flags(index) /*| Qt::ItemIsEditable*/;
    case 1:
        return QAbstractTableModel::flags(index);
    }

    return QAbstractTableModel::flags(index);
}

bool ModelCImage::insertRows(int row, int count, const QModelIndex &parent)
{
    beginInsertRows(QModelIndex(), row, row+count-1);
    endInsertRows();
    return true;
}

bool ImagesSFModel::filterAcceptsRow(int sourceRow, const QModelIndex &sourceParent) const
{

    QModelIndex index1 = sourceModel()->index(sourceRow, 1, sourceParent);

    QString strColl_1 = sourceModel()->data(index1).toString();

    if( strColl_1 == "")
        return false;
    else
        return !strColl_1.contains(HORSIMAGE);

}

bool PointGlobalSFModel::filterAcceptsRow(int sourceRow, const QModelIndex &sourceParent) const
{
    if(mAppli())
    {
        cQT_Interface * interf = (cQT_Interface*) mAppli()->Interface();
        bool saisieBasc = interf->getQTWinMode() == BASC;

        if(((int)mAppli()->PG().size() == sourceRow) && !saisieBasc)
            return false;

        ModelPointGlobal*   model   = (ModelPointGlobal*)sourceModel();
        QModelIndex         index1  = model->index(sourceRow, 0, sourceParent);
        QString             namePG  = model->data(index1).toString();
        cSP_PointGlob *     pg      = mAppli()->PGlobOfNameSVP(namePG.toStdString());

        if(pg && sourceRow < (int)mAppli()->PG().size())
        {
            if(!pg->PG()->Disparu().IsInit())
                return true;
            else if(!pg->PG()->Disparu().Val())
                return false;
        }
        else if(saisieBasc && sourceRow >= (int)mAppli()->PG().size() && !model->caseIsSaisie(sourceRow))
            return true;
        else if(sourceRow > (int)mAppli()->PG().size() && !model->caseIsSaisie(sourceRow))
            return true;
    }

    return false;
}

cAppli_SaisiePts *PointGlobalSFModel::mAppli() const
{
    return ((ModelPointGlobal*)(sourceModel()))->getMAppli();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


ModelObjects::ModelObjects(QObject *parent, cAppli_SaisiePts *appli)
    :QAbstractTableModel(parent),
      _interface((cQT_Interface*)appli->Interface())
{

}

int ModelObjects::rowCount(const QModelIndex & /*parent*/) const
{
    return _interface->getData()->getNbPolygons();
}

int ModelObjects::columnCount(const QModelIndex &parent) const
{
    return 2;
}

QVariant ModelObjects::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (role == Qt::DisplayRole)
    {
        if (orientation == Qt::Horizontal)
        {
            switch (section)
            {
            case 0:
                return QString(tr("Object"));
            case 1:
                return QString(tr("State"));
            }
        }
    }
    return QVariant();
}

QVariant ModelObjects::data(const QModelIndex &index, int role) const
{
    if (role == Qt::DisplayRole || role == Qt::EditRole)
    {
        int aK = index.row();

        if(aK < _interface->getData()->getNbPolygons())
        {
            cPolygon* aPoly = _interface->getData()->getPolygon(aK);

            switch (index.column())
            {
                case 0:
                {
                    if ((aK == 0) && (aPoly->getMaxSize() == 2))        return QString(tr("Line"));
                    else if ((aK ==1) && (aPoly->getMaxSize() == 1))    return QString(tr("Origin"));
                    else if ((aK ==2) && (aPoly->getMaxSize() == 2))    return QString(tr("Scale"));
                    else if  (aPoly->getMaxSize() == 4)                 return QString(tr("Box 2D"));
                }
                case 1:
                {
                    if (aPoly->getMaxSize() == aPoly->size())
                        return QString(tr("Done"));
                    else
                        return QString(tr("To do"));
                }
            }
        }
    }
    return QVariant();
}

bool ModelObjects::insertRows(int row, int count, const QModelIndex &parent)
{
    beginInsertRows(QModelIndex(), row, row+count-1);
    endInsertRows();
    return true;
}

bool ObjectsSFModel::filterAcceptsRow(int sourceRow, const QModelIndex &sourceParent) const
{
    //TODO:
    return true;
}
