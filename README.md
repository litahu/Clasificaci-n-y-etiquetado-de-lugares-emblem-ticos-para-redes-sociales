# 🗺 Clasificación y etiquetado de puntos de referencia locales para redes sociales

A los servicios de almacenamiento y uso compartido de fotos les gusta tener datos de ubicación para cada foto que se sube. Con los datos de ubicación, estos servicios pueden crear características avanzadas, como la sugerencia automática de etiquetas relevantes u organización automática de fotos, que ayudan a proporcionar una experiencia de usuario atractiva.
Dada una imagen, la aplicación predice los lugares más probables donde se tomó la imagen. 

## Problema
Aunque la ubicación de una foto a menudo se puede obtener mirando los metadatos de la foto, muchas fotos subidas a estos servicios no tendrán metadatos de ubicación disponibles. Esto puede ocurrir cuando, por ejemplo, la cámara que captura la imagen no tiene GPS o si los metadatos de una foto están borrados debido a preocupaciones de privacidad.

## Objetivo
- Crear una aplicación impulsada por CNN para predecir automáticamente la ubicación de la imagen real proporcionada por el usuario.

<p align="center">
    <kbd> <img width="900" alt="jkhjk" src= "https://github.com/litahu/Clasificaci-n-y-etiquetado-de-lugares-emblem-ticos-para-redes-sociales/blob/main/static_images/sample_landmark_output.png" > </kbd> <br>
    Image — Ejemplo del Output del proyecto
</p>


## Instrucciones del proyecto

### Procedimientos iniciales
1. Abrir un terminal y clonar el repositorio, luego navegar a la carpeta descargada:
	
	```	
		git clone https://github.com/udacity/cd1821-CNN-project-starter.git
		cd cd1821-CNN-project-starter
	```
    
2. Crear un nuevo entorno de conda con python 3.7.6:

    ```
        conda create --name udacity_cnn_project -y python=3.7.6
        conda activate udacity_cnn_project
    ```
    
    NOTA: tendrá que ejecutar `condaactivate udacity_cnn_project` para cada nueva sesión de terminal.
    
3. Instalar los requisitos del proyecto:

    ```
        pip install -r requirements.txt
    ```

### Desarrollo del proyecto
Ahora que tiene un entorno de trabajo, realice los siguientes pasos:

1. Abre el `cnn_from_scratch.ipynb` notebook y sigue las instrucciones
2. Abre `transfer_learning.ipynb` y sigue las instrucciones
3. Abre `app.ipynb` y sigue las instrucciones ahí


## Información del conjunto de datos
The landmark images are a subset of the Google Landmarks Dataset v2.
