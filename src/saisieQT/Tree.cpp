#include "Tree.h"

#define HORSIMAGE "hors Image"

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
                    return QString(tr("Non saisi"));
                }
            }
        }
    }

    if (role == Qt::BackgroundColorRole)
    {
        QColor selectPGlob  = QColor("#ffa02f");
        if(mAppli->PGlob(index.row()) == _interface->currentPGlobal() && _interface->currentPGlobal())
                return selectPGlob;
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
      _interface((cQT_Interface*)appli->Interface()),
      idGlobSelect(-1)
{
}

int ModelCImage::rowCount(const QModelIndex & /*parent*/) const
{
    return mAppli->images().size();
}

int ModelCImage::columnCount(const QModelIndex & /*parent*/) const
{
    return 3;
}

QVariant ModelCImage::data(const QModelIndex &index, int role) const
{

    if (role == Qt::DisplayRole || role == Qt::EditRole)
    {
        if(index.row() < (int)mAppli->images().size())
        {
            cImage* iImage = mAppli->image(index.row());

            switch (index.column())
            {
            case 0:
                return QString("%1").arg(iImage->Name().c_str());
            case 1:
            {
                if(idGlobSelect < 0 || idGlobSelect >= (int)mAppli->PG().size())
                    return QString("");

                cSP_PointGlob* pg   = mAppli->PGlob(idGlobSelect);

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
                                return QString(tr("a valide"));
                            else
                                return QString(tr(HORSIMAGE));
                        }
                        case eEPI_Refute:
                            return QString(tr("refute"));
                        case eEPI_Douteux:
                            return QString(tr("douteux"));
                        case eEPI_Valide:
                            return QString(tr("valide"));
                        case eEPI_NonValue:
                            return QString(tr("non V"));
                        case eEPI_Disparu:
                            return QString(tr("Disparu"));
                        case eEPI_Highlight:
                            return QString(tr("highlight"));
                        }
                    }
                }

                return QString(tr(HORSIMAGE));
            }
            case 2:
            {
                if(idGlobSelect < 0 || idGlobSelect >= (int)mAppli->PG().size())
                    return QString("");

                cSP_PointGlob* pg = mAppli->PGlob(idGlobSelect);

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
        QColor NonSaisie    = QColor("#93751e");
        QColor Douteux      = QColor("#a95b3b");
        QColor Valide       = QColor("#3c7355");
        QColor imageVisible = QColor("#3a819c");
        QColor selectPGlob  = QColor("#ffa02f");

        if(idGlobSelect < 0 || idGlobSelect >= (int)mAppli->PG().size())
            return QVariant(QColor("#5f5f5f"));

        cImage* iImage = mAppli->image(index.row());

        if(index.column() == 0)
        {
            if(iImage == _interface->currentCImage() )
                return selectPGlob;
            else if (_interface->isDisplayed(iImage))
                return imageVisible;
        }

        cSP_PointGlob* pg = mAppli->PGlob(idGlobSelect);

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

    if (role == Qt::TextColorRole && index.column() == 0)
        if(!(idGlobSelect < 0 || idGlobSelect >= (int)mAppli->PG().size()))
            if(index.row() < (int)mAppli->images().size() && mAppli->image(index.row()) == _interface->currentCImage() )
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
int ModelCImage::getIdGlobSelect() const
{
    return idGlobSelect;
}

void ModelCImage::setIdGlobSelect(int value)
{
    idGlobSelect = value;
}

bool ImagesSortFilterProxyModel::filterAcceptsRow(int sourceRow, const QModelIndex &sourceParent) const
{

    QModelIndex index1 = sourceModel()->index(sourceRow, 1, sourceParent);

    QRegExp regExp(HORSIMAGE);

    return !sourceModel()->data(index1).toString().contains(regExp);

}

bool PointGlobalSortFilterProxyModel::filterAcceptsRow(int sourceRow, const QModelIndex &sourceParent) const
{
    if(mAppli())
    {
        if((int)mAppli()->PG().size() == sourceRow)
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
        else if(sourceRow > (int)mAppli()->PG().size() && !model->caseIsSaisie(sourceRow))
            return true;
    }

    return false;
}

cAppli_SaisiePts *PointGlobalSortFilterProxyModel::mAppli() const
{
    return ((ModelPointGlobal*)(sourceModel()))->getMAppli();
}

cQT_Interface *PointGlobalSortFilterProxyModel::interface()
{
    return  (cQT_Interface *)mAppli()->Interface();
}
