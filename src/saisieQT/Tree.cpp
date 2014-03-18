#include "Tree.h"

ModelPointGlobal::ModelPointGlobal(QObject *parent, cAppli_SaisiePts *appli)
    :QAbstractTableModel(parent),
      mAppli(appli)
{
}

int ModelPointGlobal::rowCount(const QModelIndex & /*parent*/) const
{
    return CountPG_CaseName();
}

int ModelPointGlobal::columnCount(const QModelIndex & /*parent*/) const
{
    return 2;
}

QVariant ModelPointGlobal::data(const QModelIndex &index, int role) const
{
    if (role == Qt::DisplayRole || role == Qt::EditRole)
    {
        if(index.row() < CountPG())
        {
            std::vector< cSP_PointGlob * > vPG = mAppli->PG();
            cSP_PointGlob * pg = vPG[index.row()];
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
        else if (index.row() < CountPG_CaseName())
        {
            int id = index.row() - CountPG();
            if(id >= 0 && id < CountCaseNamePoint())
            {
                cCaseNamePoint cnPt = mAppli->Interface()->GetCaseNamePoint(id);
                switch (index.column())
                {
                case 0:
                    return QString("%1").arg(cnPt.mName.c_str());
                case 1:
                {
                    return QString("Non saisi");
                }
                }
            }
        }
    }

    return QVariant();
}

QVariant ModelPointGlobal::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (role == Qt::DisplayRole)
    {
        if (orientation == Qt::Horizontal) {
            switch (section)
            {
            case 0:
                return QString("Point");
            case 1:
                return QString("Coordinates");
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
        if(index.row() < CountPG())
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

int ModelPointGlobal::CountPG_CaseName() const
{
    return  CountPG() + CountCaseNamePoint();
}

int ModelPointGlobal::CountPG() const
{
    return  mAppli->PG().size();
}

int ModelPointGlobal::CountCaseNamePoint() const
{
    return  mAppli->Interface()->GetNumCaseNamePoint();
}

bool ModelPointGlobal::caseIsSaisie(int idRow)
{
    int idCase = idRow - CountPG();

    QString nameCase(mAppli->Interface()->GetCaseNamePoint(idCase).mName.c_str());

    for (int i = 0; i < CountPG(); ++i)
    {
       if(nameCase == QString(mAppli->PGlob(i)->PG()->Name().c_str()))
           return true;
    }

    return false;
}
