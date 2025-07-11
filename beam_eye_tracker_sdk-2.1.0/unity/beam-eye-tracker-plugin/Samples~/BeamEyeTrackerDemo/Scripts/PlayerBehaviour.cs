using UnityEngine;
using UnityEngine.InputSystem;

public class PlayerBehaviour : MonoBehaviour
{
    // Movement settings
    public float moveSpeed = 5f; // Walking speed
    public float gravity = -9.81f; // Gravity force
    public float rotationSpeed = 90f; // Rotation speed in degrees per second
    public CharacterController controller; // Reference to CharacterController component

    private Vector3 velocity; // Tracks falling velocity (for gravity)
    private PlayerControls controls; // Input Actions class
    private Vector2 moveInput; // WASD input

    void Awake()
    {
        // Initialize Input Actions (only need Move action)
        controls = new PlayerControls();
        controls.Player.Move.performed += ctx => moveInput = ctx.ReadValue<Vector2>();
        controls.Player.Move.canceled += ctx => moveInput = Vector2.zero;
    }

    void Start()
    {
        // Ensure CharacterController is assigned
        if (controller == null)
            controller = GetComponent<CharacterController>();

        // Check if CharacterController is still missing
        if (controller == null)
        {
            Debug.LogError("CharacterController is missing on " + gameObject.name);
            enabled = false;
            return;
        }
    }

    void OnEnable()
    {
        controls.Player.Enable();
    }

    void OnDisable()
    {
        controls.Player.Disable();
    }

    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            Application.Quit();

#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#endif
        }

        if (controller == null) return;

        // --- Rotation (Left/Right) ---
        float rotationInput = moveInput.x; // A/D input (-1 for A, 1 for D)
        float rotationAmount = rotationInput * rotationSpeed * Time.deltaTime;
        transform.Rotate(Vector3.up * rotationAmount);

        // --- Movement (Up/Down) ---
        Vector3 move = transform.forward * moveInput.y;
        controller.Move(move * moveSpeed * Time.deltaTime);

        // --- Gravity ---
        if (!controller.isGrounded)
        {
            velocity.y += gravity * Time.deltaTime;
        }
        else
        {
            velocity.y = -2f; // Small downward force to stay grounded
        }
        controller.Move(velocity * Time.deltaTime);
    }
}