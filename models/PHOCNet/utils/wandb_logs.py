import wandb
from torchvision import transforms as T
from PIL import ImageDraw, ImageFont
from torchvision import transforms


def train_log(loss, total_example_ct):
    wandb.log({"loss": loss}, step=total_example_ct)
    print(f"Loss after {str(total_example_ct).zfill(5)} examples: {loss:.3f}")

def train_test_log(loss_test, loss_train, accuracy_test, accuracy_train, edit_test, edit_train, epoch):
    wandb.log({"Epoch": epoch, 
               "Train loss": loss_train, "Test loss": loss_test,
               "Train accuracy": accuracy_train, "Test accuracy": accuracy_test,
               "Train edit": edit_train, "Test edit": edit_test,
               })
    print(f"Train Loss: {loss_train:.3f}\nTest Loss: {loss_test:.3f}")

def log_images(images, predicted_labels, text_labels, epoch, mode):
    t = transforms.Compose([transforms.Normalize(0, 1/0.1),transforms.Normalize(-0.5, 1)])
    t_images = t(images)
    images_with_labels = draw_images(t_images, text_labels, predicted_labels)
    wandb.log({f"Epoch{epoch}-{mode}": [wandb.Image(im) for im in images_with_labels]})

def draw_images(images, text_labels, predicted_labels):
    transform = T.ToPILImage()
    images = [draw_one_image(transform(im), t_lab, p_lab) for im, t_lab, p_lab in zip(images, text_labels, predicted_labels)]
    return images

def draw_one_image(image, text_label, predicted_label):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    if text_label == predicted_label:
        color = "green"
    else:
        color = "red"
    text = text_label + "\n" + predicted_label
    font = ImageFont.truetype(f'', 10)
    draw.text((0,0), text, font=font, fill = color)
    return image

def lr_log(lr, epoch):
    wandb.log({"learning-rate": lr}, step=epoch)