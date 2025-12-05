<?php
session_start();
if (!isset($_SESSION['name']) || $_SESSION['module'] !== 'algebra') {
    header('Location: http://www.bitn.me/clases.php');
    exit();}
?>
<!DOCTYPE html>
<html>
<head lang="en">
    <meta charset="UTF-8">

    <!--Page Title-->
    <title>Math!</title>

    <!--Meta Keywords and Description-->
    <meta name="keywords" content="">
    <meta name="description" content="">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no"/>

    <!--Favicon-->
    <link rel="shortcut icon" href="images/math.png" title="Favicon"/>

    <!-- Main CSS Files -->
    <link rel="stylesheet" href="css/style.css">

    <!-- Namari Color CSS -->
    <link rel="stylesheet" href="css/namari-color.css">

    <!--Icon Fonts - Font Awesome Icons-->
    <link rel="stylesheet" href="css/font-awesome.min.css">

    <!-- Animate CSS-->
    <link href="css/animate.css" rel="stylesheet" type="text/css">

    <!--Google Webfonts-->
    <link href='https://fonts.googleapis.com/css?family=Open+Sans:400,300,600,700,800' rel='stylesheet' type='text/css'>

    <style type="text/css">
    .text-center {
        text-align: center;}
    .container {    
    display: flex; !important;
    justify-content: center; !important;}
    .col-2.icon-block {
      border: 2px solid black; /* Establece un borde sólido de 2px en color negro */
      border-radius: 5px; /* Agrega esquinas redondeadas */
      padding: 10px; /* Añade un relleno interno de 10px */
      cursor: pointer; /* Cambia el cursor al pasar sobre el elemento */
      transition: all 0.3s ease; /* Agrega una transición suave a todos los cambios */
    }
    /* Cambia el color del borde al pasar el ratón sobre el elemento */
    .col-2.icon-block:hover {
      border-color: gray;
    }
    #justificado {text-align: justify;}
    #justificado2{            
        text-align: justify;
        margin-left: 20px; /* Margen a la izquierda */
        margin-right: 20px; /* Margen a la derecha */
    }    
    #centrado {text-align: center;}
    #derecha {text-align: right;}
    }
        .button-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 10px;
        } 
        .subtema-button {
            background-color: #82E0AA;
            border: none;
            color: white;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 12px;
            transition: background-color 0.3s ease;
        }
        .subtema-button:hover {
            background-color: #322E2D;
        }
        .subtema-button-2 {
            background-color: #636260;
            border: none;
            color: white;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 12px;
            transition: background-color 0.3s ease;
        }
        .subtema-button-2:hover {
            background-color: #322E2D;
        }
        #justificado a {
            color: blue;
            font-weight: bold;
            white-space: nowrap;  /* Evita el salto de línea */
            text-transform: none; /* Evita que el texto esté en mayúsculas */
            display: inline;      /* Asegura que el enlace se comporte como un elemento en línea */
        }
    </style>

    <script type="text/javascript" async
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjs"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

</head>
<body>

<!-- Preloader -->
<div id="preloader">
    <div id="status" class="la-ball-triangle-path">
        <div></div>
        <div></div>
        <div></div>
    </div>
</div>
<!--End of Preloader-->

<div class="page-border" data-wow-duration="0.7s" data-wow-delay="0.2s">
    <div class="top-border wow fadeInDown animated" style="visibility: visible; animation-name: fadeInDown;"></div>
    <div class="right-border wow fadeInRight animated" style="visibility: visible; animation-name: fadeInRight;"></div>
    <div class="bottom-border wow fadeInUp animated" style="visibility: visible; animation-name: fadeInUp;"></div>
    <div class="left-border wow fadeInLeft animated" style="visibility: visible; animation-name: fadeInLeft;"></div>
</div>

<div id="wrapper">

    <header id="banner" class="scrollto clearfix" data-enllax-ratio=".5">
        <div id="header" class="nav-collapse">
            <div class="row clearfix">
                <div class="col-1">

                    <!--Main Navigation-->
                    <nav id="nav-main">
                        <ul>
                            <li>
                                <a href="/clases.php">Inicio</a>
                            </li>
                            <li>
                                <a href="#clients">Álgebra</a>
                            </li>
                        </ul>
                    </nav>
                    <!--End of Main Navigation-->

                    <div id="nav-trigger"><span></span></div>
                    <nav id="nav-mobile"></nav>

                </div>
            </div>
        </div><!--End of Header-->

    </header>

    <!--Main Content Area-->
    <main id="content">

        <!--Calculadoras-->
        <section id="clients" class="scrollto clearfix">
            <div class="row clearfix">
                <div class="col-3">

                    <div class="section-heading">
                        <h3>Matemáticas</h3>
                        <h2 class="section-title">Álgebra</h2>
                        <div class="button-container">
                            <a href="" class="subtema-button">Matrices y determinantes</a>
                            <a href="/ecuacionesalgebraicas.php" class="subtema-button">Ecuaciones algebraicas</a>
                            <a href="/sistemaseclineales.php" class="subtema-button">Sistemas de ecuaciones lineales</a> 
                            <a href="/metresolsisteclineales.php" class="subtema-button">Métodos de resolución de sistemas de ecuaciones lineales</a>
                            <a href="/espaciosvectoriales.php" class="subtema-button">Espacios vectoriales</a>
                            <a href="/aplicacioneslineales.php" class="subtema-button">Aplicaciones lineales</a>
                            <a href="/pca.php" class="subtema-button">PCA: análisis de componentes principales</a>
                            <a href="/svd.php" class="subtema-button">SVD: descomposición en valores singulares</a>
                            <a href="/algebra_evaluaciones.php" class="subtema-button-2">Evaluaciones</a>
                        </div>
                    </div>

                </div>

            <div class="col-2-3">


<!-- Copia todo esto dentro de la etiqueta <body> de tu página HTML -->
<section id="tema-matrices">
  <h1>Tema 1 — Matrices y determinantes</h1>
  <p>Este tema presenta una introducción exhaustiva pensada para estudiantes de primer semestre de ingeniería. Incluye definiciones formales, propiedades, ejemplos resueltos y fragmentos de código en <code>Python</code> que puedes ejecutar en clase o dejar como prácticas.</p>

  <article id="1-1">
    <h2>1.1 Operaciones con matrices</h2>
    <h4>Definición</h4>
    <p>Una <strong>matriz</strong> es un arreglo rectangular de números con <em>m</em> filas y <em>n</em> columnas. La denotamos como A ∈ ℝ<sup>m×n</sup> y sus entradas como a<sub>ij</sub> (fila i, columna j).</p>

    <h4>Operaciones básicas</h4>
    <br>
    <ul>
      <li><strong>Suma:</strong> (A + B)<sub>ij</sub> = a<sub>ij</sub> + b<sub>ij</sub> (solo si A y B tienen misma forma).</li>
      <li><strong>Multiplicación por escalar:</strong> (αA)<sub>ij</sub> = α a<sub>ij</sub>.</li>
      <li><strong>Producto matricial:</strong> (AB)<sub>ij</sub> = Σ<sub>k=1</sub><sup>p</sup> a<sub>ik</sub> b<sub>kj</sub>, si A es m×p y B es p×n.</li>
      <li><strong>Transpuesta:</strong> A<sup>T</sup> transforma filas en columnas: (A<sup>T</sup>)<sub>ij</sub> = a<sub>ji</sub>.</li>
      <li><strong>Conjugada (para matrices complejas):</strong> Ā = conjugado de cada entrada; A* o Aᴴ denota la transpuesta conjugada.</li>
    </ul>
    <br>
    <h4>Ejemplo práctico en Python (numpy)</h4>
    <br>
    <pre><code class="language-python"># Operaciones básicas con matrices (numpy)
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 3]])
print('A =\n', A)
print('B =\n', B)
print('\nA + B =\n', A + B)
print('\n2 * A =\n', 2 * A)
print('\nA @ B (producto matricial) =\n', A @ B)  # o np.dot(A,B)
print('\nA.T =\n', A.T)
</code></pre>
    <br>
    <h4>Interpretación geométrica (2×2)</h4>
    <p>Una matriz 2×2 actúa como una transformación lineal del plano: rota, escala y/o deforma vectores. Veremos un ejemplo gráfico más abajo.</p>
  </article>

  <article id="1-2">

    <h2>1.2 Tipos de matrices</h2>
    
    <ul>
      <li><strong>Matriz cuadrada:</strong> m = n.</li>
      <li><strong>Matriz fila / columna:</strong> 1×n o n×1.</li>
      <li><strong>Matriz diagonal:</strong> todas las entradas fuera de la diagonal principal son cero.</li>
      <li><strong>Matriz identidad:</strong> I<sub>n</sub> con 1 en la diagonal principal y 0 en otros lugares.</li>
      <li><strong>Matriz triangular (inferior/superior):</strong> ceros por encima o por debajo de la diagonal.</li>
      <li><strong>Matriz simétrica:</strong> A = A<sup>T</sup>.</li>
      <li><strong>Matriz ortogonal:</strong> A<sup>T</sup>A = I (transformación que preserva longitudes).</li>
    </ul>

    <br>
    <h4>Ejemplos (Python)</h4>
    <br>
    <pre><code class="language-python">import numpy as np
I3 = np.eye(3)             # identidad 3x3
D = np.diag([2, 3, 4])     # matriz diagonal
U = np.triu(np.arange(1,10).reshape(3,3))  # triangular superior (ejemplo)
print('I3 =\n', I3)
print('D =\n', D)
print('U =\n', U)
</code></pre>
  </article>

  <article id="1-3">
    <h2>1.3 Cálculo de determinantes: desarrollo por filas y por columnas</h2>
    <h4>Determinante (intuitivo)</h4>
    <p>El <strong>determinante</strong> de una matriz cuadrada A (denotado det(A)) mide el factor de escala de la transformación lineal asociada. Para 2×2: det([[a,b],[c,d]]) = ad - bc.</p>

    <h4>Desarrollo por cofactores (expansión por filas o columnas)</h4>
    <p>Sea A una matriz n×n. Para una fila i fija:</p>
    <blockquote>det(A) = Σ<sub>j=1</sub><sup>n</sup> (-1)<sup>i+j</sup> a<sub>ij</sub> det(M<sub>ij</sub>)</blockquote>
    <p>donde M<sub>ij</sub> es la submatriz que resulta al eliminar la fila i y columna j (menor).</p>

    <h4>Ejemplo: 3×3</h4>
    <p>Para A = [[a,b,c],[d,e,f],[g,h,i]]:</p>
    <blockquote>det(A) = a·det([[e,f],[h,i]]) - b·det([[d,f],[g,i]]) + c·det([[d,e],[g,h]])</blockquote>

    <br>
    <h4>Implementación educativa (recursiva) en Python</h4>
    <br>
    <pre><code class="language-python"># Cálculo recursivo del determinante (útil para enseñar; no eficiente para n grande)
def det_recursive(M):
    # M: lista de listas o numpy array convertible a lista
    M = [list(row) for row in M]
    n = len(M)
    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0]*M[1][1] - M[0][1]*M[1][0]
    total = 0
    for j in range(n):
        # construir menor M_0j (eliminar fila 0 y columna j)
        minor = [row[:j] + row[j+1:] for row in M[1:]]
        cofactor = ((-1)**(0 + j)) * M[0][j]
        total += cofactor * det_recursive(minor)
    return total

# Ejemplo
A = [[2, 3, 1], [4, 5, 6], [7, 8, 9]]
print('det(A) recursivo =', det_recursive(A))

# Para uso práctico y eficiente, usar numpy.linalg.det
import numpy as np
print('det(A) con numpy =', np.linalg.det(np.array(A)))
</code></pre>

    <br>
    <h4>Precaución</h4>
    <p>Para matrices grandes el método recursivo es ineficiente (factorial). Use bibliotecas numéricas (LU, SVD) para cálculos reales.</p>
  </article>

  <article id="1-4">
    <h2>1.4 Inversa de una matriz</h2>
    <h4>Definición</h4>
    <p>Una matriz cuadrada A (n×n) es invertible si existe B tal que AB = BA = I. En este caso B = A<sup>-1</sup>.</p>

    <h4>Condición</h4>
    <p>A es invertible ⇔ det(A) ≠ 0.</p>

    <h4>Cálculo mediante la adjunta (teórico)</h4>
    <p>A<sup>-1</sup> = (1/det(A)) · adj(A), donde adj(A) es la traspuesta de la matriz de cofactores.</p>

    <h4>Implementación en Python</h4>
    <br>
    <pre><code class="language-python">import numpy as np
A = np.array([[4, 7], [2, 6]], dtype=float)
print('A =\n', A)
print('det(A) =', np.linalg.det(A))
A_inv = np.linalg.inv(A)
print('\nA^{-1} =\n', A_inv)

# Verificación
print('\nA @ A^{-1} =\n', A @ A_inv)
</code></pre>

    <br>
    <h4>Ejemplo pedagógico: inversa por adjunta (para 2x2)</h4>
    <br>
    <pre><code class="language-python"># Inversa por fórmula para 2x2
import numpy as np
A = np.array([[a, b], [c, d]], dtype=float)  # sustituye a,b,c,d por números
detA = a*d - b*c
if detA == 0:
    print('No invertible')
else:
    A_inv = (1/detA) * np.array([[d, -b], [-c, a]])
    print('A^{-1} =\n', A_inv)
</code></pre>

    <br>
    <h4>Advertencia numérica</h4>
    <p>En cálculos numéricos reales es preferible resolver sistemas (por ejemplo, usar factorización LU) en vez de calcular la inversa explícita cuando solo quieres resolver Ax=b.</p>
  </article>

  <article id="1-5">
    <h2>1.5 Rango de una matriz</h2>
    <h4>Definición</h4>
    <p>El <strong>rango</strong> de A (rank(A)) es el número máximo de columnas (o filas) linealmente independientes. Mide la dimensión de la imagen de la transformación lineal asociada.</p>

    <h4>Propiedades</h4>
    <br>
    <ul>
      <li>rank(A) ≤ min(m, n).</li>
      <li>Si A es invertible (n×n) entonces rank(A) = n.</li>
      <li>rank(A) = rank(A<sup>T</sup>).</li>
    </ul>

    <br>
    <h4>Cálculo numérico (Python)</h4>
    <br>
    <pre><code class="language-python">import numpy as np
A = np.array([[1,2,3],[2,4,6],[3,6,9]], dtype=float)
print('A =\n', A)
print('rank(A) =', np.linalg.matrix_rank(A))
</code></pre>

    <br>
    <h4>Ejemplo interpretativo</h4>
    <p>La matriz del ejemplo tiene filas dependientes (fila2 = 2·fila1, fila3 = 3·fila1) por lo que rank = 1. Geométricamente, la transformación colapsa el espacio a una recta.</p>
  </article>

  <article id="graficas">
    <h2>Visualizaciones y ejemplos aplicados</h2>

    <h4>Transformación lineal en 2D (gráfica)</h4>
    <p>Ejecuta el siguiente código en Spyder para ver cómo una matriz 2×2 transforma el cuadrado unitario y los vectores base.</p>

    <pre><code class="language-python"># Visualizar transformación lineal 2x2
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[1.5, -0.5], [0.5, 1]])  # ejemplo de transformación
square = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]]).T  # 2x5
transformed = A @ square

plt.figure()
plt.plot(square[0,:], square[1,:], '-o', label='Cuadrado original')
plt.plot(transformed[0,:], transformed[1,:], '-o', label='Cuadrado transformado')

# vectores base antes y después
e1 = np.array([1,0]); e2 = np.array([0,1])
Ae1 = A @ e1; Ae2 = A @ e2
plt.arrow(0,0,e1[0],e1[1], head_width=0.05, color='gray')
plt.arrow(0,0,e2[0],e2[1], head_width=0.05, color='gray')
plt.arrow(0,0,Ae1[0],Ae1[1], head_width=0.05, color='blue')
plt.arrow(0,0,Ae2[0],Ae2[1], head_width=0.05, color='blue')

plt.legend()
plt.axis('equal')
plt.grid(True)
plt.title('Transformación lineal: acción de una matriz 2x2')
plt.show()
</code></pre>
  </article>

  <footer>
    <h3>Actividades sugeridas</h3>
    <ol>
      <li>Calcula determinantes 2×2 y 3×3 a mano y verifica con numpy.</li>
      <li>Implementa la función recursiva del determinante y compárala con numpy.linalg.det en tiempos para n=2..7.</li>
      <li>Resuelve sistemas Ax=b usando la inversa y usando np.linalg.solve; compara resultados y tiempos.</li>
      <li>Explora el rango de matrices construidas aleatoriamente y explica por qué la probabilidad de que una matriz cuadrada aleatoria sea singular es prácticamente 0 (en teoría, depende del dominio).</li>
    </ol>

    <p>— <strong>Prof. Eder Díaz</strong></p>
  </footer>
</section>


            </div>

            </div>
        </section>
        <!--End of Calculadoras-->

    </main>
    <!--End Main Content Area-->

</div>

<!-- Include JavaScript resources -->
<script src="js/jquery.1.8.3.min.js"></script>
<script src="js/wow.min.js"></script>
<script src="js/featherlight.min.js"></script>
<script src="js/featherlight.gallery.min.js"></script>
<script src="js/jquery.enllax.min.js"></script>
<script src="js/jquery.scrollUp.min.js"></script>
<script src="js/jquery.easing.min.js"></script>
<script src="js/jquery.stickyNavbar.min.js"></script>
<script src="js/jquery.waypoints.min.js"></script>
<script src="js/images-loaded.min.js"></script>
<script src="js/lightbox.min.js"></script>
<script src="js/site.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://kit.fontawesome.com/30c014f357.js" crossorigin="anonymous"></script>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

</body>
</html>