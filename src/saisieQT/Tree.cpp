#include "Tree.h"

TreeItem::TreeItem(const QList<QVariant> &data, TreeItem *parent)
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

QVariant TreeItem::data(int column) const
{
    return itemData.value(column);
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

TreeModel::TreeModel(QObject *parent):
    QAbstractItemModel(parent)
{
    QList<QVariant> rootData;
    rootData << "Point" << "Image" << "State" << "Coordinates";
    rootItem = new TreeItem(rootData);
}

TreeModel::~TreeModel()
{
    delete rootItem;
}

int TreeModel::columnCount(const QModelIndex &parent) const
{
    if (parent.isValid())
        return static_cast<TreeItem*>(parent.internalPointer())->columnCount();
    else
        return rootItem->columnCount();
}

void TreeModel::setAppli(cAppli_SaisiePts *appli)
{
    _appli = appli;

    setupModelData(rootItem);
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

    return QAbstractItemModel::flags(index) | Qt::ItemIsEditable;
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
         if (index.row() == 0) //point name
         {
             std::string oldName = index.data(Qt::DisplayRole).toString().toStdString();

             _appli->ChangeName(oldName,value.toString().toStdString());

             emit dataChanged();

             return true;
         }
     }
     return false;
 }

QModelIndex TreeModel::index(int row, int column, const QModelIndex &parent)
            const
{
    if (!hasIndex(row, column, parent))
        return QModelIndex();

    TreeItem *parentItem;

    if (!parent.isValid())
        parentItem = rootItem;
    else
        parentItem = static_cast<TreeItem*>(parent.internalPointer());

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

    TreeItem *childItem = static_cast<TreeItem*>(index.internalPointer());
    TreeItem *parentItem = childItem->parent();

    if (parentItem == rootItem)
        return QModelIndex();

    return createIndex(parentItem->row(), 0, parentItem);
}

int TreeModel::rowCount(const QModelIndex &parent) const
{
    TreeItem *parentItem;
    if (parent.column() > 0)
        return 0;

    if (!parent.isValid())
        parentItem = rootItem;
    else
        parentItem = static_cast<TreeItem*>(parent.internalPointer());

    return parentItem->childCount();
}

void TreeModel::setupModelData(TreeItem *parent)
{
   // cout << "setting model data" << endl;
    QList<TreeItem*> parents;
    parents << parent;

    std::vector<cSP_PointGlob *> vPts = _appli->PG();

    for (int bK=0; bK < (int) vPts.size(); ++bK)
    {
        QList<QVariant> columnData;

        cSP_PointGlob* aPG = vPts[bK];

        // Point global

        QString namePt  = QString(aPG->PG()->Name().c_str());
        QString coord3d = QString::number(aPG->PG()->P3D().Val().x, 'f' ,1) + " " +
                          QString::number(aPG->PG()->P3D().Val().y, 'f' ,1) + " " +
                          QString::number(aPG->PG()->P3D().Val().z, 'f' ,1);

        columnData << namePt << "" << "" << coord3d;

        TreeItem * item = new TreeItem(columnData, parent);
        parents.last()->appendChild(item);

        //Pointes image

        std::map<std::string,cSP_PointeImage *> map = aPG->getPointes();

        std::map<std::string,cSP_PointeImage *>::const_iterator it = map.begin();

        for(; it != map.end(); ++it)
        {
             QList<QVariant> columnData2;

             std::string     aName = it->first;
             cSP_PointeImage* aPIm = it->second;

             cOneSaisie* aSom = aPIm->Saisie();

             QString nameImg = QString(aName.c_str());

             QString state   = StateToQString(aSom->Etat());
             QString coord2d = QString::number(aSom->PtIm().x, 'f' ,1) + " " +
                                   QString::number(aSom->PtIm().y, 'f' ,1);

             columnData2 << "" << nameImg << state << coord2d;

             item->appendChild(new TreeItem(columnData2, item));
        }
    }
}

