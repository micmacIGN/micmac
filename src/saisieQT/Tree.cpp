#include "Tree.h"

TreeItem::TreeItem(const QVector<QVariant> &data, TreeItem *parent)
{
    parentItem = parent;
    itemData = data;
}

TreeItem::~TreeItem()
{
    qDeleteAll(childItems);
}

void TreeItem::appendChild(TreeItem *item)
{
    childItems.append(item);
}

TreeItem *TreeItem::child(int row)
{
    return childItems.value(row);
}

int TreeItem::childCount() const
{
    return childItems.count();
}

int TreeItem::columnCount() const
{
    return itemData.count();
}

int TreeItem::childNumber() const
{
    if (parentItem)
        return parentItem->childItems.indexOf(const_cast<TreeItem*>(this));

    return 0;
}

QVariant TreeItem::data(int column) const
{
    return itemData.value(column);
}

bool TreeItem::setData(int column, const QVariant &value)
{
    if (column < 0 || column >= itemData.size())
           return false;

    itemData[column] = value;
    return true;
}

TreeItem *TreeItem::parent()
{
    return parentItem;
}

int TreeItem::row() const
{
    if (parentItem)
        return parentItem->childItems.indexOf(const_cast<TreeItem*>(this));

    return 0;
}

bool TreeItem::insertChildren(int position, int count, int columns)
{
    if (position < 0 || position > childItems.size())
        return false;

    for (int row = 0; row < count; ++row)
    {
        QVector<QVariant> data(columns);
        TreeItem *item = new TreeItem(data, this);
        childItems.insert(position, item);
    }

    return true;
}

bool TreeItem::removeChildren(int position, int count)
{
    if (position < 0 || position + count > childItems.size())
        return false;

    for (int row = 0; row < count; ++row)
        delete childItems.takeAt(position);

    return true;
}

TreeModel::TreeModel(QObject *parent):
    QAbstractItemModel(parent)
{
    QVector<QVariant> rootData;
    rootData << "Point" << "Image" << "State" << "Coordinates";
    rootItem = new TreeItem(rootData);
}

TreeModel::~TreeModel()
{
    delete rootItem;
}

TreeItem *TreeModel::getItem(const QModelIndex &index) const
{
    if (index.isValid())
    {
        TreeItem *item = static_cast<TreeItem*>(index.internalPointer());
        if (item)
            return item;
    }
    return rootItem;
}

int TreeModel::columnCount(const QModelIndex &/*parent*/) const
{
    return rootItem->columnCount();
}

void TreeModel::setAppli(cAppli_SaisiePts *appli)
{
    _appli = appli;
}

QString StateToQString(eEtatPointeImage state)
{
    switch (state)
    {
    case eEPI_NonSaisi:
        return "Non Saisi";
    case eEPI_Refute:
        return "Refute";
    case eEPI_Douteux:
        return "Douteux";
    case eEPI_Valide:
        return "Valide";
    case eEPI_NonValue:
        return "NonValue";
    case eEPI_Disparu:
        return "Disparu";
    case eEPI_Highlight:  //only for QT interface
    case eEPI_Deleted:    //only for QT interface
        return "";
    }

    return "";
}

QVariant TreeModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid())
        return QVariant();

    if (role == Qt::DisplayRole || role == Qt::EditRole)
    {
        TreeItem *item = static_cast<TreeItem*>(index.internalPointer());

        return item->data(index.column());
    }
    else
        return QVariant();
}

Qt::ItemFlags TreeModel::flags(const QModelIndex &index) const
{
    if (!index.isValid())
        return 0;

    return QAbstractItemModel::flags(index) | Qt::ItemIsEditable | Qt::ItemIsSelectable | Qt::ItemIsEnabled;
}

QVariant TreeModel::headerData(int section, Qt::Orientation orientation,
                               int role) const
{
    if (orientation == Qt::Horizontal && role == Qt::DisplayRole)
        return rootItem->data(section);

    return QVariant();
}

bool TreeModel::setData(const QModelIndex &index,
                        const QVariant &value, int role)
 {
     if (index.isValid() && role == Qt::EditRole)
     {
         if (index.column() == 0) //point name
         {
             std::string oldName = index.data(Qt::DisplayRole).toString().toStdString();

             _appli->ChangeName(oldName,value.toString().toStdString());

             TreeItem *Item = static_cast<TreeItem*>(index.internalPointer());

             if (Item->setData(0, value))
                 emit dataChanged(index, index);

             return true;
         }
     }
     return false;
 }

QModelIndex TreeModel::index(int row, int column, const QModelIndex &parent) const
{
    if (parent.isValid() && parent.column() != 0)
        return QModelIndex();

    TreeItem *parentItem = getItem(parent);

    TreeItem *childItem = parentItem->child(row);
    if (childItem)
        return createIndex(row, column, childItem);
    else
        return QModelIndex();
}

QModelIndex TreeModel::parent(const QModelIndex &index) const
{
    if (!index.isValid())
        return QModelIndex();

    TreeItem *childItem = getItem(index);
    TreeItem *parentItem = childItem->parent();

    if (parentItem == rootItem)
        return QModelIndex();

    return createIndex(parentItem->childNumber(), 0, parentItem);
}

int TreeModel::rowCount(const QModelIndex &parent) const
{
    TreeItem *parentItem = getItem(parent);

    return parentItem->childCount();
}

bool TreeModel::insertRows(int position, int rows, const QModelIndex &parent)
{
    TreeItem *parentItem = getItem(parent);
    bool success;

    beginInsertRows(parent, position, position + rows - 1);
    success = parentItem->insertChildren(position, rows, rootItem->columnCount());
    endInsertRows();

    return success;
}

bool TreeModel::removeRows(int position, int rows, const QModelIndex &parent)
{
    TreeItem *parentItem = getItem(parent);
    bool success = true;

    beginRemoveRows(parent, position, position + rows - 1);
    success = parentItem->removeChildren(position, rows);
    endRemoveRows();

    return success;
}


QVector <QVariant> TreeModel::buildRow(cSP_PointGlob* aPG)
{
    QVector<QVariant> columnData;

    QString namePt  = QString(aPG->PG()->Name().c_str());
    QString coord3d = QString::number(aPG->PG()->P3D().Val().x, 'f' ,1) + " " +
                      QString::number(aPG->PG()->P3D().Val().y, 'f' ,1) + " " +
                      QString::number(aPG->PG()->P3D().Val().z, 'f' ,1);

    columnData << namePt << "" << "" << coord3d;

    return columnData;
}

QVector <QVariant> TreeModel::buildChildRow(std::pair < std::string , cSP_PointeImage*> data)
{
    QVector <QVariant> columnData;

    std::string     aName = data.first;
    cSP_PointeImage* aPIm = data.second;

    cOneSaisie* aSom = aPIm->Saisie();

    QString nameImg = QString(aName.c_str());

    QString state   = StateToQString(aSom->Etat());
    QString coord2d = QString::number(aSom->PtIm().x, 'f' ,1) + " " +
                      QString::number(aSom->PtIm().y, 'f' ,1);

    columnData << "" << nameImg << state << coord2d;

    return columnData;
}

int TreeModel::getColumnSize(int column, QFontMetrics fm)
{
    int colWidth = -1;
    for (int aK=0; aK < rowCount();++aK)
    {
        QModelIndex id = index(aK, column);

        QString text = data(id, Qt::DisplayRole).toString();

        int textWidth = fm.width(text);

        if (colWidth < textWidth) colWidth = textWidth;
    }

    return colWidth;
}

void TreeModel::addPoint(cSP_PointeImage * aPIm)
{
    int pos= _appli->PG().size()-1;
    cSP_PointGlob* aPG;

    if (aPIm)
    {
        aPG = aPIm->Gl();
    }
    else
    {
        aPG = _appli->PG().back();
    }

    //check if point already exists
    string name = aPG->PG()->Name();

    QModelIndex id;
    for (int aK=0; aK < rowCount(rootItem->index());++aK)
    {
        QString text = data(index(aK, 0), Qt::DisplayRole).toString();

        if (text.toStdString() == name)
        {
            id = index(aK, 0);
            pos = aK;
        }
    }

    if ( !id.isValid() ) //add point
    {
        pos = rowCount(rootItem->index());


        QModelIndex idx = index(pos-1, 0);

        if (!insertRow(idx.row()+1, idx.parent()))
            return;

        QModelIndex newIdx = index(pos, 0);

        TreeItem * item = static_cast<TreeItem*>(newIdx.internalPointer());

        if (item)
        {
            item->setData(buildRow(aPG));

            std::map<std::string,cSP_PointeImage *> map = aPG->getPointes();

            std::map<std::string,cSP_PointeImage *>::const_iterator it = map.begin();

            for(; it != map.end(); ++it)
            {
                item->appendChild(new TreeItem(buildChildRow(*it), item));
            }
        }
    }
    else //insert infos
    {
        TreeItem * item = static_cast<TreeItem*>(id.internalPointer());

        if (item)
        {
            item->setData(buildRow(aPG));

            std::map<std::string,cSP_PointeImage *> map = aPG->getPointes();

            std::map<std::string,cSP_PointeImage *>::const_iterator it = map.begin();

            for(; it != map.end(); ++it)
            {
                item->appendChild(new TreeItem(buildChildRow(*it), item));
            }
        }
    }
}

void TreeModel::setupModelData()
{
    QList<TreeItem*> parents;
    parents << rootItem;

    std::vector<cSP_PointGlob *> vPts = _appli->PG();
    for (int bK=0; bK < (int) vPts.size(); ++bK)
    {
        cSP_PointGlob* aPG = vPts[bK];

        TreeItem * item = new TreeItem(buildRow(aPG), rootItem);
        parents.last()->appendChild(item);

        //Pointes image

        std::map<std::string,cSP_PointeImage *> map = aPG->getPointes();

        std::map<std::string,cSP_PointeImage *>::const_iterator it = map.begin();

        for(; it != map.end(); ++it)
        {
            item->appendChild(new TreeItem(buildChildRow(*it), item));
        }
    }

    for (int aK=0; aK < _appli->Interface()->GetNumCaseNamePoint(); ++aK)
    {
        QVector<QVariant> columnData;

        columnData << QString(_appli->Interface()->GetCaseNamePoint(aK).mName.c_str()) << "" << "" << "";

        TreeItem * item = new TreeItem(columnData, rootItem);
        parents.last()->appendChild(item);
    }
}

void TreeModel::setPointGlob(QModelIndex idx, cSP_PointGlob* aPG)
{
    TreeItem * item = static_cast<TreeItem*>(idx.internalPointer());

    if (item)
    {
        item->setData(buildRow(aPG));

        //Pointes image

        std::map<std::string,cSP_PointeImage *> map = aPG->getPointes();

        std::map<std::string,cSP_PointeImage *>::const_iterator it = map.begin();

        int bK = 0;
        for(; it != map.end(); ++it, ++bK)
        {
            item->child(bK)->setData(buildChildRow(*it));
        }
    }
}

void TreeModel::updateData()
{
    std::vector<cSP_PointGlob *> vPts = _appli->PG();

    std::vector <std::string> ptNames;

    for (int aK=0; aK < (int) vPts.size(); ++aK)
    {
        cSP_PointGlob* aPG = vPts[aK];

        std::map<std::string,cSP_PointeImage *> map = aPG->getPointes();

        std::map<std::string,cSP_PointeImage *>::const_iterator it = map.begin();

        int nbDisparus = 0;
        for(; it != map.end(); ++it)
        {
            cSP_PointeImage* aPIm = it->second;

            if (aPIm->Saisie()->Etat() == eEPI_Disparu)
                ++nbDisparus;
        }

        if (nbDisparus == (int) map.size())
        {
            ptNames.push_back(aPG->PG()->Name());
        }
        else
        {
            for (int bK=0; bK < rowCount(rootItem->index());++bK) //TODO: factoriser
            {
                QString text = data(index(bK, 0), Qt::DisplayRole).toString();

                if (text.toStdString() == aPG->PG()->Name())
                {
                    setPointGlob(index(bK, 0), aPG);
                }
            }
        }
    }

    QVector <QModelIndex> vIdx;
    for (int aK =0; aK< (int) ptNames.size(); ++aK)
    {
        for (int bK=0; bK < rowCount(rootItem->index());++bK)
        {
            QString text = data(index(bK, 0), Qt::DisplayRole).toString();

            if (text.toStdString() == ptNames[aK])
            {
                vIdx.insert(0, index(bK, 0));
            }
        }
    }

    for (int aK=0; aK < vIdx.size(); ++aK)
    {
        removeRow(vIdx[aK].row(), vIdx[aK].parent());
    }
}

