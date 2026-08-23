import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import os
import pandas as pd
import shutil

# ── Config ───────────────────────────────────────────────────────────────
RAW_IMAGES_DIR = 'data/images'
DATA_DIR       = 'data/dataset'
STYLES_CSV     = 'data/styles.csv'
MODEL_PATH     = 'CNN_Model.pth'
BATCH_SIZE     = 32
EPOCHS         = 10
LR             = 0.001
IMG_SIZE       = 224
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Mapping Myntra article types → our 10 classes ────────────────────────
LABEL_MAP = {
    # T-shirt/top
    'Tshirts'          : 'T-shirt',
    'Tops'             : 'T-shirt',
    'Tank Tops'        : 'T-shirt',
    'Tunics'           : 'T-shirt',
    # Trouser
    'Trousers'         : 'Trouser',
    'Jeans'            : 'Trouser',
    'Shorts'           : 'Trouser',
    'Leggings'         : 'Trouser',
    'Jeggings'         : 'Trouser',
    'Capris'           : 'Trouser',
    'Track Pants'      : 'Trouser',
    # Pullover
    'Sweaters'         : 'Pullover',
    'Sweatshirts'      : 'Pullover',
    'Jackets'          : 'Pullover',
    'Waistcoat'        : 'Pullover',
    'Patiala'          : 'Pullover',
    # Dress
    'Dresses'          : 'Dress',
    'Skirts'           : 'Dress',
    'Sarees'           : 'Dress',
    'Salwar'           : 'Dress',
    'Kurta Sets'       : 'Dress',
    'Dupatta'          : 'Dress',
    # Coat
    'Coats'            : 'Coat',
    'Blazers'          : 'Coat',
    'Suits'            : 'Coat',
    'Shrug'            : 'Coat',
    # Sandal
    'Sandals'          : 'Sandal',
    'Flats'            : 'Sandal',
    'Heels'            : 'Sandal',
    'Flip Flops'       : 'Sandal',
    # Shirt
    'Shirts'           : 'Shirt',
    'Kurtas'           : 'Shirt',
    'Formal Shirts'    : 'Shirt',
    'Casual Shirts'    : 'Shirt',
    # Sneaker
    'Casual Shoes'     : 'Sneaker',
    'Sports Shoes'     : 'Sneaker',
    'Sneakers'         : 'Sneaker',
    'Running Shoes'    : 'Sneaker',
    # Bag
    'Handbags'         : 'Bag',
    'Backpacks'        : 'Bag',
    'Clutches'         : 'Bag',
    'Wallets'          : 'Bag',
    'Trolley Bag'      : 'Bag',
    'Messenger Bag'    : 'Bag',
    # Ankle boot
    'Boots'            : 'Ankle boot',
    'Ankle'            : 'Ankle boot',
    'Chelsea Boots'    : 'Ankle boot',
}


# ── Step 1: Organize images into class folders ───────────────────────────
def organize_dataset():
    if not os.path.exists(STYLES_CSV):
        print(f"ERROR: {STYLES_CSV} not found.")
        print("Download styles.csv from kaggle.com/datasets/paramaggarwal/fashion-product-images-small")
        print("and place it in the data/ folder.")
        return False

    print("Reading styles.csv...")
    df = pd.read_csv(STYLES_CSV, on_bad_lines='skip')
    print(f"Total entries: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    os.makedirs(DATA_DIR, exist_ok=True)
    copied, skipped = 0, 0

    for _, row in df.iterrows():
        try:
            img_id    = str(int(row['id']))
            article   = str(row['articleType']).strip()
            mapped    = LABEL_MAP.get(article)
            if mapped is None:
                skipped += 1
                continue

            src = os.path.join(RAW_IMAGES_DIR, f"{img_id}.jpg")
            if not os.path.exists(src):
                skipped += 1
                continue

            dst_dir = os.path.join(DATA_DIR, mapped)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(src, os.path.join(dst_dir, f"{img_id}.jpg"))
            copied += 1
        except Exception:
            skipped += 1
            continue

    print(f"\nDataset organized: {copied} images copied, {skipped} skipped")
    print("\nClass distribution:")
    for cls in sorted(os.listdir(DATA_DIR)):
        count = len(os.listdir(os.path.join(DATA_DIR, cls)))
        print(f"  {cls}: {count} images")
    return True


# ── Step 2: Transforms ───────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225]),
])


# ── Step 3: MobileNetV2 ──────────────────────────────────────────────────
def build_model(num_classes):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, num_classes)
    )
    return model


# ── Step 4: Train ────────────────────────────────────────────────────────
def train():
    if not os.path.exists(DATA_DIR):
        success = organize_dataset()
        if not success:
            return

    print(f"\nUsing device: {DEVICE}")

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    print(f"Total images : {len(full_dataset)}")
    print(f"Classes      : {full_dataset.classes}")

    val_size   = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = build_model(len(full_dataset.classes)).to(DEVICE)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            correct += (outputs.argmax(1) == labels).sum().item()
            total   += labels.size(0)
        train_acc = 100 * correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total   += labels.size(0)
        val_acc = 100 * val_correct / val_total
        scheduler.step()

        print(f"Epoch [{epoch+1}/{EPOCHS}]  Train: {train_acc:.1f}%  Val: {val_acc:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes'         : full_dataset.classes,
                'img_size'        : IMG_SIZE,
            }, MODEL_PATH)
            print(f"  --> Saved best model (val: {val_acc:.1f}%)")

    print(f"\nDone. Best val accuracy: {best_acc:.1f}%")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == '__main__':
    train()
